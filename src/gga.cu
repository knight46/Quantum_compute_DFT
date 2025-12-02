/*
 * gga.cu – PBE-GGA (Final Corrections)
 * 1. Precision: Aligned PBE Correlation with Libxc to fix 0.24Ha error.
 * 2. Stability: Added limits for small/large gradients in core regions.
 * 3. Layout: Planar memory layout for all kernels.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <Eigen/Dense>

#define DEBUG_GGA_CORR 0

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::RowMajor;

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); exit(1); } }while(0)

__device__ inline double atomicAdd_double(double *address,double val){
#if __CUDA_ARCH__>=600
    return atomicAdd(address,val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do { assumed = old; old = atomicCAS(address_as_ull, assumed, __double_as_longlong(__longlong_as_double(assumed) + val)); } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// Lower epsilon to capture core/tail contributions better
#define RHO_EPS 1e-20
#define MIN_GRAD 1e-28

// Standard PW92/PBE Constants
__constant__ double A_pw92 = 0.03109069086965489503; 
__constant__ double alpha1 = 0.21370;
__constant__ double beta1  = 7.5957;
__constant__ double beta2  = 3.5876;
__constant__ double beta3  = 1.6382;
__constant__ double beta4  = 0.49294;

/* ---------- Device Functions ---------- */

// PW92 Correlation (RKS) - Standard
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

// PBE Exchange - Libxc Aligned
__device__ inline void pbe_exchange(double rho, double sigma, double &ex, double &vrho, double &vsigma){
    if(rho < RHO_EPS) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
    
    const double Cx = -0.7385587663820224; 
    const double kappa = 0.804;
    const double mu = 0.2195149727645171; 
    
    double rho13 = pow(rho, 1.0/3.0);
    double rho43 = rho * rho13;
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    
    // s = |grad rho| / (2 kF rho)
    // s2 = sigma / (4 kF^2 rho^2)
    double s2 = 0.0;
    // Robust s2 calculation
    if (sigma > MIN_GRAD) {
        double denom = 4.0 * kF * kF * rho * rho;
        if(denom > 1e-50) s2 = sigma / denom;
    }
    
    // Libxc clamping
    if(s2 > 1e12) s2 = 1e12; 

    double num = 1.0 + mu * s2 / kappa;
    double den = 1.0; 
    // F = 1 + kappa - kappa / (1 + mu s2 / kappa)
    double F = 1.0 + kappa * (1.0 - 1.0/num);
    
    ex = Cx * rho13 * F; 
    
    // Derivatives
    // dF/ds2 = mu / (1 + mu s2/kappa)^2
    double dF_ds2 = mu / (num * num);
    
    // vsigma = d(rho ex)/dsigma
    //        = rho * (Cx rho^1/3) * dF/ds2 * ds2/dsigma
    // ds2/dsigma = 1 / (4 kF^2 rho^2)
    vsigma = (Cx * rho43) * dF_ds2 * (1.0 / (4.0 * kF * kF * rho * rho));
    
    // vrho = Ex + rho * dEx/drho
    // dEx/drho = 4/3 Cx rho^1/3 F + Cx rho^4/3 dF/ds2 * ds2/drho
    // ds2/drho = s2 * (-8/3) / rho
    vrho = (4.0/3.0) * ex - (8.0/3.0) * (Cx * rho43) * s2 * dF_ds2 / rho;
}

// PBE Correlation - Libxc Aligned (Critical Fixes)
__device__ inline void pbe_correlation(double rho, double sigma, double &ec, double &vrho, double &vsigma){
    if(rho < RHO_EPS) { ec=0.0; vrho=0.0; vsigma=0.0; return; }

    // 1. LDA part
    double ec_lda, vc_lda;
    pw92_correlation_rks(rho, ec_lda, vc_lda);

    const double beta = 0.066725;
    const double gamma = 0.03109069086965489503; 

    // 2. Reduced gradient t
    // t^2 = sigma * pi / (16 * kF * rho^2)
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    double t2 = 0.0;
    if(sigma > MIN_GRAD) {
        double denom = 16.0 * kF * rho * rho;
        if(denom > 1e-50) t2 = (sigma * M_PI) / denom;
    }
    if(t2 > 1.0e20) t2 = 1.0e20;

    // 3. H calculation
    double x = -ec_lda / gamma;
    // Careful expm1 for numerical stability near 0
    double expm1_x = expm1(x);
    double A = 0.0;
    if(fabs(expm1_x) < 1e-20) A = 1.0e20; // Avoid div zero
    else A = (beta/gamma) / expm1_x;

    double At2 = A * t2;
    double num = 1.0 + At2;
    double den = 1.0 + At2 + At2*At2;
    double Q = num / den;
    
    double term_log = 1.0 + (beta/gamma) * t2 * Q;
    double H = gamma * log(term_log);
    
    ec = ec_lda + H;

    // 4. Derivatives for Potential
    // dQ/d(At2)
    double Q_prime = (den - num * (1.0 + 2.0*At2)) / (den * den);
    
    double pre_log = gamma / term_log * (beta/gamma);
    double dH_dt2 = pre_log * (Q + At2 * Q_prime);
    double dH_dA  = pre_log * t2 * t2 * Q_prime;

    // vsigma
    // dt2/dsigma = pi / (16 kF rho^2)
    double dt2_dsig = 0.0;
    double denom_sig = 16.0 * kF * rho * rho;
    if(denom_sig > 1e-50) dt2_dsig = M_PI / denom_sig;
    
    vsigma = rho * dH_dt2 * dt2_dsig;

    // vrho
    // dA/dx = -A * exp(x) / (exp(x) - 1)
    double exp_x = exp(x);
    double dA_dx = -A * exp_x / expm1_x;
    // dx/drho = (vc_lda - ec_lda) / (rho * gamma)
    double dx_drho = (vc_lda - ec_lda) / (rho * gamma); // Note order: vc - ec
    double dA_drho = dA_dx * dx_drho;
    
    // dt2/drho = t2 * (-7/3) / rho
    double dt2_drho = t2 * (-7.0/3.0) / rho;
    
    vrho = vc_lda + H + rho * (dH_dA * dA_drho + dH_dt2 * dt2_drho);
}

/* ---------- Kernels (Planar) ---------- */

__global__ void gga_exc_vxc_kernel(int ngrid, const double *rho, const double *sigma, double *exc, double *vrho, double *vsigma){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=ngrid) return;
    // if(g == 1000) {
    //     printf("DEBUG GPU [Grid 1000]: Rho = %.8e, Sigma = %.8e\n", rho[g], sigma[g]);
    // }
    
    double r_val=rho[g];
    double s_val=sigma[g];

    double ex=0, vrx=0, vsx=0;
    double ec=0, vrc=0, vsc=0;
    // if(g == 13000) {
    //     printf("DEBUG INTEGRATOR [Grid 13000]: Rho=%.4f, Sigma=%.4e (Expect ~3.2e5)\n", r_val, s_val);
    // }
    // Only compute if density is significant
    if(r_val > RHO_EPS) {
        pbe_exchange(r_val, s_val, ex, vrx, vsx);
        pbe_correlation(r_val, s_val, ec, vrc, vsc);
    }
    
    // exc store: total energy density per unit volume = rho * epsilon
    if(exc)    exc[g]    = r_val * (ex + ec); 
    if(vrho)   vrho[g]   = vrx + vrc;
    if(vsigma) vsigma[g] = vsx + vsc;
}

// Planar Density & Gradient
__global__ void get_rho_sigma_kernel_planar(int nao, int rows,
                                            const double *dm, 
                                            const double *ao, 
                                            const double *gx, 
                                            const double *gy, 
                                            const double *gz, 
                                            double *rho, double *sigma, double *grad_rho_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows) return;

    const double *phi = ao + (size_t)g * nao;
    const double *gphi_x = gx + (size_t)g * nao;
    const double *gphi_y = gy + (size_t)g * nao;
    const double *gphi_z = gz + (size_t)g * nao;

    double r = 0.0;
    double gr_x = 0.0, gr_y = 0.0, gr_z = 0.0;

    for (int u = 0; u < nao; ++u) {
        double phiu = phi[u];
        double dx_u = gphi_x[u];
        double dy_u = gphi_y[u];
        double dz_u = gphi_z[u];

        const double *dm_row = dm + (size_t)u * nao;

        for (int v = 0; v < nao; ++v) {
            double dm_val = dm_row[v];
            double phiv = phi[v];
            
            r += dm_val * phiu * phiv;
            
            // 梯度计算公式
            double term_x = dx_u * phiv + phiu * gphi_x[v];
            double term_y = dy_u * phiv + phiu * gphi_y[v];
            double term_z = dz_u * phiv + phiu * gphi_z[v];

            gr_x += dm_val * term_x;
            gr_y += dm_val * term_y;
            gr_z += dm_val * term_z;
        }
    }

    rho[g] = r;
    double s = gr_x*gr_x + gr_y*gr_y + gr_z*gr_z;
    sigma[g] = s;
    
    // --- 核心区域探针 ---
    // 只有当密度足够大（原子核附近）且是第一个线程时打印，避免刷屏
    // if (r > 1.0 && g % 1000 == 0) {
    //      // 检查梯度是否异常小
    //      printf("DEBUG CORE [Grid %d]: Rho=%.4f, Sigma=%.4e. Grad=(%.2e, %.2e, %.2e). gphi_x[0]=%.2e\n", 
    //             g, r, s, gr_x, gr_y, gr_z, gphi_x[0]);
    // }
    // ------------------

    if(grad_rho_out){
        grad_rho_out[g*3+0] = gr_x;
        grad_rho_out[g*3+1] = gr_y;
        grad_rho_out[g*3+2] = gr_z;
    }
}

// Planar Vxc Matrix (Symmetrized)
__global__ void build_vxc_matrix_gga_kernel_planar(
    int nao, int rows, int g0,
    const double *ao_b,
    const double *gx_b, 
    const double *gy_b, 
    const double *gz_b, 
    const double *w_b,
    const double *vrho_b,
    const double *vsigma_b,
    const double *grad_rho_b, 
    double *vxc_mat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= rows * nao) return;
    
    int im = idx / nao; 
    int i  = idx % nao; 

    double w = w_b[im];
    if(fabs(w) < 1e-15) return;

    double vr = vrho_b[im];
    double vs = vsigma_b[im];
    
    double gr_x = grad_rho_b[im*3+0];
    double gr_y = grad_rho_b[im*3+1];
    double gr_z = grad_rho_b[im*3+2];

    double aoi = ao_b[im*nao+i];
    double dxi = gx_b[im*nao+i];
    double dyi = gy_b[im*nao+i];
    double dzi = gz_b[im*nao+i];

    double dot_rho_phi_i = gr_x * dxi + gr_y * dyi + gr_z * dzi;

    for(int j=i; j<nao; ++j){
        double aoj = ao_b[im*nao+j];
        double dxj = gx_b[im*nao+j];
        double dyj = gy_b[im*nao+j];
        double dzj = gz_b[im*nao+j];

        double dot_rho_phi_j = gr_x * dxj + gr_y * dyj + gr_z * dzj;

        double term_rho = vr * aoi * aoj;
        double term_sig = 2.0 * vs * (aoi * dot_rho_phi_j + aoj * dot_rho_phi_i);
        
        double val = w * (term_rho + term_sig);

        atomicAdd_double(vxc_mat + i*nao+j, val);
        if (i != j) {
            atomicAdd_double(vxc_mat + j*nao+i, val);
        }
    }
}

// Integration Kernel
__global__ void weighted_sum_gga_kernel(const double *w,const double *exc,double *out,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    atomicAdd_double(out,w[tid]*exc[tid]);
}

// ... (build_coulomb_kernel & get_rho_kernel remain unchanged)
__global__ void build_coulomb_kernel(int nao, int rows_m, int m0, const double *eri_slice, const double *dm, double *J) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = rows_m * nao;
    if (idx >= tot) return;
    int im = idx / nao; int n = idx % nao; int m = m0 + im;
    double sum = 0.0;
    for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
            size_t pos = ((size_t)im * nao + n) * nao * nao + (size_t)l * nao + s;
            sum += dm[l * nao + s] * eri_slice[pos];
        }
    }
    atomicAdd_double(&J[m * nao + n], sum);
}
__global__ void get_rho_kernel(int nao, int ngrid, const double *dm, const double *ao, double *rho_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;
    const double *phi_g = ao + (size_t)g * nao;
    double r = 0.0;
    for (int u = 0; u < nao; ++u) {
        double phiu = phi_g[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) r += dm_row[v] * phiu * phi_g[v];
    }
    rho_out[g] = r;
}

/* ---------- Host Functions ---------- */

extern "C" {

// Vxc Matrix Host
void build_vxc_matrix_gga(int nao, int ngrid, 
                          const double *ao, 
                          const double *ao_grad, // PySCF Planar (3, N, NAO)
                          const double *weights,
                          const double *rho, const double *sigma, const double *grad_rho, 
                          double *vxc_mat) 
{
    size_t free_byte, total_byte; 
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE = size_t(free_byte * 0.9); 
    const size_t aux = 64 << 20;
    const size_t per_row = (nao * 4 + 7) * sizeof(double); 
    size_t rows_per = (SAFE > aux) ? (SAFE - aux) / per_row : 0; 
    if(rows_per == 0) exit(1); 
    if(rows_per > ngrid) rows_per = ngrid;

    double *d_ao=0, *d_grad=0, *d_w=0, *d_rho=0, *d_sig=0, *d_grho=0, *d_vr=0, *d_vs=0, *d_mat=0;
    CUDA_CHECK(cudaMalloc(&d_ao, rows_per*nao*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per*3*nao*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_w, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grho, rows_per*3*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_vr, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_vs, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mat, (size_t)nao*nao*sizeof(double))); 
    CUDA_CHECK(cudaMemset(d_mat, 0, (size_t)nao*nao*sizeof(double)));

    double *d_gx = d_grad;
    double *d_gy = d_grad + rows_per * nao;
    double *d_gz = d_grad + 2 * rows_per * nao;
    const int BLOCK = 256;
    
    for(int g0=0; g0<ngrid; g0+=rows_per){
        int g1 = std::min(g0+(int)rows_per, ngrid); 
        int rows = g1 - g0;
        size_t copy_size_ao = rows * nao * sizeof(double);

        CUDA_CHECK(cudaMemcpyAsync(d_ao, ao + g0*nao, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_w, weights + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho, rho + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig, sigma + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_grho, grad_rho + g0*3, rows*3*sizeof(double), cudaMemcpyHostToDevice));

        // Planar Gradients
        const double *src_x = ao_grad + (size_t)0 * ngrid * nao + (size_t)g0 * nao;
        const double *src_y = ao_grad + (size_t)1 * ngrid * nao + (size_t)g0 * nao;
        const double *src_z = ao_grad + (size_t)2 * ngrid * nao + (size_t)g0 * nao;

        CUDA_CHECK(cudaMemcpyAsync(d_gx, src_x, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gy, src_y, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gz, src_z, copy_size_ao, cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        gga_exc_vxc_kernel<<<grid, BLOCK>>>(rows, d_rho, d_sig, nullptr, d_vr, d_vs);
        
        int N = rows * nao; 
        int grid2 = (N + BLOCK - 1) / BLOCK;
        build_vxc_matrix_gga_kernel_planar<<<grid2, BLOCK>>>(
            nao, rows, g0, d_ao, d_gx, d_gy, d_gz, d_w, d_vr, d_vs, d_grho, d_mat
        );
    }
    CUDA_CHECK(cudaMemcpy(vxc_mat, d_mat, (size_t)nao*nao*sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_grho));
    CUDA_CHECK(cudaFree(d_vr)); CUDA_CHECK(cudaFree(d_vs)); CUDA_CHECK(cudaFree(d_mat));
}

// Compute Energy
double compute_exc_energy_gga(int ngrid, const double *weights, const double *rho, const double *sigma) {
    size_t free_byte,total_byte; CUDA_CHECK(cudaMemGetInfo(&free_byte,&total_byte));
    const size_t SAFE=size_t(free_byte*0.9); const size_t aux=64<<20;
    const size_t per_row=4*sizeof(double); size_t rows_per=(SAFE>aux)?(SAFE-aux)/per_row:0;
    if(rows_per==0) exit(1); if(rows_per>ngrid) rows_per=ngrid;
    double *d_w,*d_rho,*d_sig,*d_exc,*d_sum;
    CUDA_CHECK(cudaMalloc(&d_w, rows_per*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_rho, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_exc, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    const int BLOCK=256; double exc_total=0.0;
    for(int g0=0;g0<ngrid;g0+=rows_per){
        int g1=std::min(g0+(int)rows_per,ngrid); int rows=g1-g0;
        CUDA_CHECK(cudaMemcpyAsync(d_w, weights+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho, rho+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig, sigma+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        int grid=(rows+BLOCK-1)/BLOCK;
        gga_exc_vxc_kernel<<<grid,BLOCK>>>(rows,d_rho,d_sig,d_exc,nullptr,nullptr);
        CUDA_CHECK(cudaMemset(d_sum,0,sizeof(double)));
        weighted_sum_gga_kernel<<<grid,BLOCK>>>(d_w,d_exc,d_sum,rows);
        double partial=0.0; CUDA_CHECK(cudaMemcpy(&partial,d_sum,sizeof(double),cudaMemcpyDeviceToHost));
        exc_total+=partial;
    }
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig));
    CUDA_CHECK(cudaFree(d_exc)); CUDA_CHECK(cudaFree(d_sum)); return exc_total;
}

// Get Rho/Sigma (Planar Host)
void get_rho_sigma(int nao, int ngrid, const double *dm, const double *ao, const double *ao_grad, double *rho, double *sigma, double *grad_rho) 
{
    size_t free_byte, total_byte; CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE = size_t(free_byte * 0.9); 
    const size_t dm_bytes = nao * nao * sizeof(double);
    const size_t row_bytes = (4 * nao + 5) * sizeof(double); 
    const size_t aux = 64 << 20;
    size_t rows_per = (SAFE > dm_bytes + aux) ? (SAFE - dm_bytes - aux) / row_bytes : 0;
    if (rows_per == 0) exit(1); if (rows_per > ngrid) rows_per = ngrid;

    double *d_dm, *d_ao, *d_grad, *d_rho, *d_sig, *d_grho;
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes)); 
    CUDA_CHECK(cudaMalloc(&d_ao, rows_per * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per * 3 * nao * sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per * sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grho, rows_per * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));

    double *d_gx = d_grad;
    double *d_gy = d_grad + rows_per * nao;
    double *d_gz = d_grad + 2 * rows_per * nao;
    const int BLOCK = 128;
    for (int g0 = 0; g0 < ngrid; g0 += rows_per) {
        int g1 = std::min(g0 + (int)rows_per, ngrid); int rows = g1 - g0;
        size_t copy_size = rows * nao * sizeof(double);

        CUDA_CHECK(cudaMemcpyAsync(d_ao, ao + g0 * nao, copy_size, cudaMemcpyHostToDevice));
        // Planar Gradients
        const double *src_x = ao_grad + (size_t)0 * ngrid * nao + (size_t)g0 * nao;
        const double *src_y = ao_grad + (size_t)1 * ngrid * nao + (size_t)g0 * nao;
        const double *src_z = ao_grad + (size_t)2 * ngrid * nao + (size_t)g0 * nao;
        CUDA_CHECK(cudaMemcpyAsync(d_gx, src_x, copy_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gy, src_y, copy_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gz, src_z, copy_size, cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        get_rho_sigma_kernel_planar<<<grid, BLOCK>>>(nao, rows, d_dm, d_ao, d_gx, d_gy, d_gz, d_rho, d_sig, d_grho);
        CUDA_CHECK(cudaMemcpyAsync(rho + g0, d_rho, rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(sigma + g0, d_sig, rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(grad_rho + g0 * 3, d_grho, rows * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_grho));
}

// ... (build_coulomb_matrix and get_rho remain unchanged) ...
void build_coulomb_matrix(int nao, const double *eri, const double *dm, double *J) {
    size_t free,total; CUDA_CHECK(cudaMemGetInfo(&free,&total));
    size_t rows_per = (free*0.9 - nao*nao*sizeof(double)*2)/(nao*nao*nao*sizeof(double));
    if(rows_per<1) rows_per=1; if(rows_per>nao) rows_per=nao;
    double *d_dm,*d_J,*d_eri;
    CUDA_CHECK(cudaMalloc(&d_dm, nao*nao*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_J, nao*nao*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eri, rows_per*nao*nao*nao*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, nao*nao*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J, 0, nao*nao*sizeof(double)));
    for(int m0=0;m0<nao;m0+=rows_per){
        int m1=std::min(m0+(int)rows_per,nao); int rows=m1-m0;
        CUDA_CHECK(cudaMemcpyAsync(d_eri, eri+m0*nao*nao*nao, rows*nao*nao*nao*sizeof(double),cudaMemcpyHostToDevice));
        int grid=((rows*nao)+255)/256;
        build_coulomb_kernel<<<grid,256>>>(nao,rows,m0,d_eri,d_dm,d_J);
    }
    CUDA_CHECK(cudaMemcpy(J,d_J,nao*nao*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_J)); CUDA_CHECK(cudaFree(d_eri));
}

void get_rho(int nao, int ngrid, const double *dm, const double *ao, double *rho_out) {
    size_t free, total; CUDA_CHECK(cudaMemGetInfo(&free, &total));
    size_t rows_per = (free * 0.9 - nao * nao * sizeof(double)) / (nao * sizeof(double) + sizeof(double));
    if (rows_per < 1) rows_per = 1; if (rows_per > ngrid) rows_per = ngrid;
    double *d_dm, *d_ao, *d_rho;
    CUDA_CHECK(cudaMalloc(&d_dm, nao * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ao, rows_per * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, nao * nao * sizeof(double), cudaMemcpyHostToDevice));
    for (int g0 = 0; g0 < ngrid; g0 += rows_per) {
        int g1 = std::min(g0 + (int)rows_per, ngrid); int rows = g1 - g0;
        CUDA_CHECK(cudaMemcpyAsync(d_ao, ao + g0 * nao, rows * nao * sizeof(double), cudaMemcpyHostToDevice));
        int grid = (rows + 127) / 128;
        get_rho_kernel<<<grid, 128>>>(nao, rows, d_dm, d_ao, d_rho);
        CUDA_CHECK(cudaMemcpyAsync(rho_out + g0, d_rho, rows * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_rho));
}

} // extern "C"

extern "C" void solve_fock_eigen(int n, const double *F_in, const double *S_in, double *e, double *C) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> F(n,n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> S(n,n);
    std::memcpy(F.data(), F_in, n*n*sizeof(double));
    std::memcpy(S.data(), S_in, n*n*sizeof(double));
    GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> solver(F, S, Eigen::ComputeEigenvectors);
    VectorXd evalues = solver.eigenvalues();
    auto evecs = solver.eigenvectors();
    std::memcpy(e, evalues.data(), n*sizeof(double));
    std::memcpy(C, evecs.data(),  n*n*sizeof(double));
}