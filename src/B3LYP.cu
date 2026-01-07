#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error at line %d\n",__LINE__); } }while(0)

#define RHO_EPS 1e-15
#define MIN_GRAD 1e-20

// --- B3LYP Parameters ---
__constant__ double C_LDA_X = 0.80;     
__constant__ double C_B88_X = 0.72;
__constant__ double C_VWN_C = 0.19;     
__constant__ double C_LYP_C = 0.81;

// VWN3 Parameters 
__constant__ double A_VWN = 0.0310907;
__constant__ double b_VWN = 13.0720;
__constant__ double c_VWN = 42.7198;
__constant__ double x0_VWN = -0.409286;

// B88 Parameters
__constant__ double BETA_B88 = 0.0042;

// LYP Parameters
__constant__ double A_LYP = 0.04918;
__constant__ double B_LYP = 0.132;
__constant__ double C_LYP = 0.2533;
__constant__ double D_LYP = 0.349;



__device__ inline void slater_exchange(double rho, double &ex, double &vx) {
    const double Cx = -0.7385587663820224; // -(3/4)*(3/pi)^(1/3)
    if (rho < RHO_EPS) { ex = 0.0; vx = 0.0; return; }
    
    double rho13 = pow(rho, 1.0/3.0);
    ex = Cx * rho13;         
    vx = (4.0/3.0) * ex;     
}

__device__ inline void b88_exchange(double rho, double sigma, double &ex, double &vrho, double &vsigma) {
    if (rho < RHO_EPS) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
    
    double rho13 = pow(rho, 1.0/3.0);
    double rho43 = rho * rho13;
    double grad_rho = sqrt(sigma);
    
    if (grad_rho < MIN_GRAD) { ex=0.0; vrho=0.0; vsigma=0.0; return; }

    double x = grad_rho / rho43;
    double x2 = x * x;
    double asinh_x = asinh(x);
    
    double denom = 1.0 + 6.0 * BETA_B88 * x * asinh_x;
    double term = BETA_B88 * x2 / denom;
    
    ex = -term * rho13; 


    double d_denom_dx = 6.0 * BETA_B88 * (asinh_x + x / sqrt(1.0 + x2));
    double dF_dx = -BETA_B88 * (2.0 * x * denom - x2 * d_denom_dx) / (denom * denom);
    

    double dE_dx = rho43 * dF_dx;
    

    vsigma = dE_dx * (x / (2.0 * sigma));
    

    double E_dens = rho43 * (-term);
    vrho = (4.0/3.0) * (E_dens / rho) - (4.0/3.0) * dE_dx * (x / rho);
}

__device__ inline void vwn_correlation(double rho, double &ec, double &vc) {
    if (rho < RHO_EPS) { ec=0.0; vc=0.0; return; }
    
    const double pi = 3.14159265358979323846;
    double rs = pow(3.0 / (4.0 * pi * rho), 1.0/3.0);
    double x = sqrt(rs);
    
    double X = x * x + b_VWN * x + c_VWN;
    double Q = sqrt(4.0 * c_VWN - b_VWN * b_VWN);
    
    double log_term = log(x * x / X);
    double atan_term = (2.0 / Q) * atan(Q / (2.0 * x + b_VWN));
    
    double x02 = x0_VWN * x0_VWN;
    double X_x0 = x02 + b_VWN * x0_VWN + c_VWN; 
    
    double term_c_log = log(pow(x - x0_VWN, 2.0) / X);
    double term_c_atan = (2.0 * (2.0 * x0_VWN + b_VWN) / Q) * atan(Q / (2.0 * x + b_VWN));
    

    double eps_c = A_VWN * ( log_term + b_VWN * atan_term - 
                             (b_VWN * x0_VWN / X_x0) * (term_c_log + term_c_atan) );
    
    ec = eps_c;


    double d_log = 2.0/x - (2.0*x + b_VWN)/X;
    

    double d_atan = -1.0 / X; 

    double d_term_c_log = 2.0/(x - x0_VWN) - (2.0*x + b_VWN)/X;
    

    double d_term_c_atan = -(2.0*x0_VWN + b_VWN) / X;

    double deps_dx = A_VWN * ( d_log + b_VWN * d_atan - 
                               (b_VWN * x0_VWN / X_x0) * (d_term_c_log + d_term_c_atan) );

    vc = eps_c - (rs / 3.0) * (deps_dx / (2.0 * x));
}

__device__ inline void lyp_correlation(double rho, double sigma, double &ec, double &vrho, double &vsigma) {
    if (rho < RHO_EPS) { ec=0.0; vrho=0.0; vsigma=0.0; return; }

    double rho_m13 = pow(rho, -1.0/3.0);
    double rho_113 = pow(rho, 11.0/3.0); 
    
    double a = A_LYP;
    double b = B_LYP;
    double c = C_LYP;
    double d = D_LYP;

    double exp_fac = exp(-c * rho_m13);
    double denom = 1.0 + d * rho_m13;
    double denom_inv = 1.0 / denom;
    
    double Cf = (3.0/10.0) * pow(3.0 * M_PI * M_PI, 2.0/3.0);
    

    double tp = (1.0/9.0) * sigma / rho; 

    
    double rho_53 = pow(rho, 5.0/3.0);
    double bracket = Cf * rho_53 - (17.0/72.0) * sigma / rho;

    double rho_m23 = pow(rho, -2.0/3.0);
    
    double term1 = -a * denom_inv; 
    double omega = -a * b * exp_fac * rho_m23 * denom_inv; 

    double E_dens = term1 * rho + omega * bracket;
    ec = E_dens / rho;


    double d_denom_drho = d * (-1.0/3.0) * pow(rho, -4.0/3.0);
    double d_term1_drho = -term1 * denom_inv * d_denom_drho; 

    double d_omega_drho = omega * ( (d_term1_drho / term1) + 
                                    (c/3.0)*pow(rho, -4.0/3.0) - 
                                    (2.0/3.0)/rho );


    double d_bracket_drho = Cf * (5.0/3.0) * pow(rho, 2.0/3.0) + (17.0/72.0) * sigma / (rho*rho);


    double d_bracket_dsigma = -(17.0/72.0) / rho;


    vrho = term1 + rho * d_term1_drho + omega * d_bracket_drho + bracket * d_omega_drho;

    vsigma = omega * d_bracket_dsigma;
}


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
            double prod_n = dm_val * phi[v]; 
            
            r += prod_n * phiu;
            
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

__global__ void b3lyp_fused_kernel(
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

    double ex_lda=0, vx_lda=0;
    slater_exchange(r_val, ex_lda, vx_lda);

    double ex_b88=0, vrx_b88=0, vsx_b88=0;
    b88_exchange(r_val, s_val, ex_b88, vrx_b88, vsx_b88);

    double ec_vwn=0, vc_vwn=0;
    vwn_correlation(r_val, ec_vwn, vc_vwn);

    double ec_lyp=0, vrc_lyp=0, vsc_lyp=0;
    lyp_correlation(r_val, s_val, ec_lyp, vrc_lyp, vsc_lyp);
    
    double total_eps = C_LDA_X * ex_lda + 
                       C_B88_X * ex_b88 + 
                       C_VWN_C * ec_vwn + 
                       C_LYP_C * ec_lyp;

    if (!compute_B && exc_out) {
        exc_out[k] = r_val * total_eps;
        return; 
    }

    if (compute_B) {
        double total_vrho = C_LDA_X * vx_lda + 
                            C_B88_X * vrx_b88 + 
                            C_VWN_C * vc_vwn + 
                            C_LYP_C * vrc_lyp;
        

        double total_vsigma = C_B88_X * vsx_b88 + 
                              C_LYP_C * vsc_lyp;

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
            B_row[i] = weight * (total_vrho * phi_row[i] + 2.0 * total_vsigma * dot);
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
    CUDA_CHECK(cudaGetLastError());
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
    
    b3lyp_fused_kernel<<<grid, block>>>(
        ngrid, nao, true, 
        d_weights, d_rho, d_sigma, d_grad_rho, 
        d_ao, d_gx, d_gy, d_gz, 
        nullptr, d_B_work
    );
    CUDA_CHECK(cudaGetLastError());

    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nao, nao, ngrid,
        &alpha,
        d_ao, nao,      
        d_B_work, nao,  
        &beta,
        d_vxc, nao      
    ));
}

double compute_exc_gpu(int ngrid, int nao, const double *d_weights, 
                       const double *d_rho, const double *d_sigma, double *d_exc_work)
{
    int block = 256;
    int grid = (ngrid + block - 1) / block;

    b3lyp_fused_kernel<<<grid, block>>>(
        ngrid, nao, false, 
        nullptr, d_rho, d_sigma, nullptr, 
        nullptr, nullptr, nullptr, nullptr, 
        d_exc_work, nullptr
    );
    CUDA_CHECK(cudaGetLastError());

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
    
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, 
                N2, N2, 
                &alpha, 
                d_eri, N2, 
                d_dm, 1,   
                &beta, 
                d_J, 1     
    ));
}

} 