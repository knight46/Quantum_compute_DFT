#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error at line %d\n",__LINE__); } }while(0)

#define RHO_EPS 1e-20
#define MIN_GRAD 1e-28

__constant__ double A_pw92 = 0.03109069086965489503; 
__constant__ double alpha1 = 0.21370;
__constant__ double beta1  = 7.5957;
__constant__ double beta2  = 3.5876;
__constant__ double beta3  = 1.6382;
__constant__ double beta4  = 0.49294;

__device__ inline void pw92_correlation_rks(double rho, double &ec, double &vc) {
    if (rho < RHO_EPS) { ec = 0.0; vc = 0.0; return; }
    const double rs = pow(3.0 / (4.0 * M_PI * rho), 1.0/3.0);
    const double rs_sqrt = sqrt(rs);
    double Q = 2.0 * A_pw92 * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs * rs_sqrt + beta4 * rs * rs);
    double Q_prime = 2.0 * A_pw92 * (0.5 * beta1 / rs_sqrt + beta2 + 1.5 * beta3 * rs_sqrt + 2.0 * beta4 * rs);
    double log_term = log(1.0 + 1.0 / Q);
    double f_rs = -2.0 * A_pw92 * (1.0 + alpha1 * rs);
    ec = f_rs * log_term;
    double df_drs = -2.0 * A_pw92 * alpha1;
    double term2 = f_rs * (1.0 / (1.0 + 1.0/Q)) * (-1.0 / (Q*Q)) * Q_prime;
    double dec_drs = df_drs * log_term + term2;
    vc = ec - (rs / 3.0) * dec_drs;
}

__device__ inline void pbe_exchange(double rho, double sigma, double &ex, double &vrho, double &vsigma){
    if(rho < RHO_EPS) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
    const double Cx = -0.7385587663820224; 
    const double kappa = 0.804;
    const double mu = 0.2195149727645171; 
    double rho13 = pow(rho, 1.0/3.0);
    double rho43 = rho * rho13;
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    double s2 = 0.0;
    if (sigma > MIN_GRAD) {
        double denom = 4.0 * kF * kF * rho * rho;
        if(denom > 1e-50) s2 = sigma / denom;
    }
    if(s2 > 1e12) s2 = 1e12; 
    double num = 1.0 + mu * s2 / kappa;
    double F = 1.0 + kappa * (1.0 - 1.0/num);
    ex = Cx * rho13 * F; 
    double dF_ds2 = mu / (num * num);
    vsigma = (Cx * rho43) * dF_ds2 * (1.0 / (4.0 * kF * kF * rho * rho));
    vrho = (4.0/3.0) * ex - (8.0/3.0) * (Cx * rho43) * s2 * dF_ds2 / rho;
}

__device__ inline void pbe_correlation(double rho, double sigma, double &ec, double &vrho, double &vsigma){
    if(rho < RHO_EPS) { ec=0.0; vrho=0.0; vsigma=0.0; return; }
    double ec_lda, vc_lda;
    pw92_correlation_rks(rho, ec_lda, vc_lda);
    const double beta = 0.066725;
    const double gamma = 0.03109069086965489503; 
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    double t2 = 0.0;
    if(sigma > MIN_GRAD) {
        double denom = 16.0 * kF * rho * rho;
        if(denom > 1e-50) t2 = (sigma * M_PI) / denom;
    }
    if(t2 > 1.0e20) t2 = 1.0e20;
    double x = -ec_lda / gamma;
    double expm1_x = expm1(x);
    double A = 0.0;
    if(fabs(expm1_x) < 1e-20) A = 1.0e20; 
    else A = (beta/gamma) / expm1_x;
    double At2 = A * t2;
    double num = 1.0 + At2;
    double den = 1.0 + At2 + At2*At2;
    double Q = num / den;
    double term_log = 1.0 + (beta/gamma) * t2 * Q;
    double H = gamma * log(term_log);
    ec = ec_lda + H;
    double Q_prime = (den - num * (1.0 + 2.0*At2)) / (den * den);
    double pre_log = gamma / term_log * (beta/gamma);
    double dH_dt2 = pre_log * (Q + At2 * Q_prime);
    double dH_dA  = pre_log * t2 * t2 * Q_prime;
    double dt2_dsig = 0.0;
    double denom_sig = 16.0 * kF * rho * rho;
    if(denom_sig > 1e-50) dt2_dsig = M_PI / denom_sig;
    vsigma = rho * dH_dt2 * dt2_dsig;
    double exp_x = exp(x);
    double dA_dx = -A * exp_x / expm1_x;
    double dx_drho = (vc_lda - ec_lda) / (rho * gamma); 
    double dA_drho = dA_dx * dx_drho;
    double dt2_drho = t2 * (-7.0/3.0) / rho;
    vrho = vc_lda + H + rho * (dH_dA * dA_drho + dH_dt2 * dt2_drho);
}

// --- Kernels ---


__global__ void get_rho_sigma_kernel_planar(int nao, int rows, const double *dm, const double *ao, 
                                            const double *gx, const double *gy, const double *gz, 
                                            double *rho, double *sigma, double *grad_rho_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows) return;
    
    const double *phi = ao + (size_t)g * nao;
    const double *gphi_x = gx + (size_t)g * nao;
    const double *gphi_y = gy + (size_t)g * nao;
    const double *gphi_z = gz + (size_t)g * nao;

    double r = 0.0, gr_x = 0.0, gr_y = 0.0, gr_z = 0.0;

    for (int u = 0; u < nao; ++u) {
        double phiu = phi[u];
        double dx_u = gphi_x[u]; 
        double dy_u = gphi_y[u]; 
        double dz_u = gphi_z[u];
        
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) {
            double dm_val = dm_row[v];
            double val = dm_val * phi[v]; 
            r += val * phiu;
            gr_x += dm_val * (dx_u * phi[v] + phiu * gphi_x[v]);
            gr_y += dm_val * (dy_u * phi[v] + phiu * gphi_y[v]);
            gr_z += dm_val * (dz_u * phi[v] + phiu * gphi_z[v]);
        }
    }
    rho[g] = r;
    sigma[g] = gr_x*gr_x + gr_y*gr_y + gr_z*gr_z;
    if(grad_rho_out){ 
        grad_rho_out[g*3+0] = gr_x; 
        grad_rho_out[g*3+1] = gr_y; 
        grad_rho_out[g*3+2] = gr_z; 
    }
}


__global__ void gga_fused_kernel(
    int rows, int nao, bool compute_B,
    const double *w, const double *rho, const double *sigma, const double *grad_rho,
    const double *ao, const double *gx, const double *gy, const double *gz,
    double *exc_out, double *B_mat
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    if (k >= rows) return;

    double r_val = rho[k];
    double s_val = sigma[k];
    
    if (r_val < RHO_EPS) {
        if (!compute_B && exc_out) exc_out[k] = 0.0;
        if (compute_B) {
            for (int i = 0; i < nao; ++i) B_mat[k * nao + i] = 0.0;
        }
        return;
    }

    double ex=0, vrx=0, vsx=0;
    double ec=0, vrc=0, vsc=0;

    pbe_exchange(r_val, s_val, ex, vrx, vsx);
    pbe_correlation(r_val, s_val, ec, vrc, vsc);

    if (!compute_B && exc_out) {
        exc_out[k] = r_val * (ex + ec);
        return; 
    }


    if (compute_B) {
        double vrho_val = vrx + vrc;
        double vsigma_val = vsx + vsc;
        double weight = w[k];

        double gr_x = grad_rho[k*3 + 0];
        double gr_y = grad_rho[k*3 + 1];
        double gr_z = grad_rho[k*3 + 2];


        const double *phi_row = ao + k * nao;
        const double *gx_row = gx + k * nao;
        const double *gy_row = gy + k * nao;
        const double *gz_row = gz + k * nao;
        double *B_row = B_mat + k * nao;

        for (int i = 0; i < nao; ++i) {
            double dot = gr_x * gx_row[i] + gr_y * gy_row[i] + gr_z * gz_row[i];
            B_row[i] = weight * (vrho_val * phi_row[i] + 4.0 * vsigma_val * dot);
        }
    }
}

__global__ void reduce_sum_kernel(const double *w, const double *val, double *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += w[i] * val[i];
    }
    atomicAdd(out, sum);
}


extern "C" {

static cublasHandle_t handle = nullptr;

void get_rho_sigma_gpu(int nao, int ngrid, 
                       const double* d_dm, const double* d_ao, const double* d_ao_grad,
                       double* d_rho, double* d_sigma, double* d_grad_rho) 
{
    const double *d_gx = d_ao_grad;
    const double *d_gy = d_ao_grad + ngrid * nao;
    const double *d_gz = d_ao_grad + 2 * ngrid * nao;
    
    int block = 128;
    int grid = (ngrid + block - 1) / block;
    get_rho_sigma_kernel_planar<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_gx, d_gy, d_gz, d_rho, d_sigma, d_grad_rho);
}

void build_vxc_gpu(int nao, int ngrid, 
                   const double *d_ao, const double *d_ao_grad, const double *d_weights,
                   const double *d_rho, const double *d_sigma, const double *d_grad_rho,
                   double *d_B_work, double *d_vxc) 
{
    if (!handle) cublasCreate(&handle);

    const double *d_gx = d_ao_grad;
    const double *d_gy = d_ao_grad + ngrid * nao;
    const double *d_gz = d_ao_grad + 2 * ngrid * nao;


    int block = 256;
    int grid = (ngrid + block - 1) / block;
    
    gga_fused_kernel<<<grid, block>>>(
        ngrid, nao, true, 
        d_weights, d_rho, d_sigma, d_grad_rho, 
        d_ao, d_gx, d_gy, d_gz, 
        nullptr, d_B_work
    );


    
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nao, nao, ngrid,
        &alpha,
        d_ao, nao,      // A: nao x ngrid (View)
        d_B_work, nao,  // B: nao x ngrid (View) -> Transpose
        &beta,
        d_vxc, nao
    );
}

double compute_exc_gpu(int ngrid, int nao, const double *d_weights, 
                       const double *d_rho, const double *d_sigma, double *d_exc_work)
{
    int block = 256;
    int grid = (ngrid + block - 1) / block;


    gga_fused_kernel<<<grid, block>>>(
        ngrid, nao, false, 
        nullptr, d_rho, d_sigma, nullptr, 
        nullptr, nullptr, nullptr, nullptr, 
        d_exc_work, nullptr
    );

    double *d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));
    reduce_sum_kernel<<<256, 256>>>(d_weights, d_exc_work, d_sum, ngrid);
    
    double val;
    cudaMemcpy(&val, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return val;
}


void build_coulomb_gpu(int nao, const double *d_eri, const double *d_dm, double *d_J) {
    if (!handle) cublasCreate(&handle);


    
    int N2 = nao * nao;
    double alpha = 1.0;
    double beta = 0.0;


    
    cublasDgemv(handle, CUBLAS_OP_N, 
                N2, N2, 
                &alpha, 
                d_eri, N2, // A
                d_dm, 1,   // x
                &beta, 
                d_J, 1     // y
    );
}

} // extern C