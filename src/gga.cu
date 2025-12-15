/*
 * gga.cu – PBE-GGA (Final Corrections & cuBLAS Optimization)
 * 1. Precision: Aligned PBE Correlation with Libxc.
 * 2. Stability: Added limits for small/large gradients.
 * 3. Performance: Replaced atomicAdd with cuBLAS GEMM for Vxc and Ddot for Exc.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h> // Added for GEMM/Ddot
#include <algorithm>
#include <Eigen/Dense>

#define DEBUG_GGA_CORR 0

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::RowMajor;

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); exit(1); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error: %d at %s:%d\n",err,__FILE__,__LINE__); exit(1); } }while(0)

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

#define RHO_EPS 1e-20
#define MIN_GRAD 1e-28

__constant__ double A_pw92 = 0.03109069086965489503; 
__constant__ double alpha1 = 0.21370;
__constant__ double beta1  = 7.5957;
__constant__ double beta2  = 3.5876;
__constant__ double beta3  = 1.6382;
__constant__ double beta4  = 0.49294;

/* ---------- Device Functions ---------- */

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

/* ---------- Kernels ---------- */

__global__ void gga_exc_vxc_kernel(int ngrid, const double *rho, const double *sigma, double *exc, double *vrho, double *vsigma){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=ngrid) return;
    
    double r_val=rho[g];
    double s_val=sigma[g];

    double ex=0, vrx=0, vsx=0;
    double ec=0, vrc=0, vsc=0;

    if(r_val > RHO_EPS) {
        pbe_exchange(r_val, s_val, ex, vrx, vsx);
        pbe_correlation(r_val, s_val, ec, vrc, vsc);
    }
    
    if(exc)    exc[g]    = r_val * (ex + ec); 
    if(vrho)   vrho[g]   = vrx + vrc;
    if(vsigma) vsigma[g] = vsx + vsc;
}

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

    if(grad_rho_out){
        grad_rho_out[g*3+0] = gr_x;
        grad_rho_out[g*3+1] = gr_y;
        grad_rho_out[g*3+2] = gr_z;
    }
}

// NEW: Helper Kernel for GEMM-based Vxc
// Prepares the Y matrix: Y_g,i = w_g * (0.5 * vrho * phi_i + 2.0 * vsigma * (grad_rho . grad_phi_i))
__global__ void prepare_gga_potentials_kernel(
    int rows, int nao,
    const double *ao,     // [rows, nao]
    const double *gx,     // [rows, nao]
    const double *gy, 
    const double *gz,
    const double *w,      // [rows]
    const double *vrho,   // [rows]
    const double *vsigma, // [rows]
    const double *grad_rho, // [rows, 3]
    double *Y_matrix)     // Output: [rows, nao]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nao) return;

    int g = idx / nao; 
    int i = idx % nao; 

    double wg = w[g];
    if (fabs(wg) < 1e-15) {
        Y_matrix[idx] = 0.0;
        return;
    }

    double vr = vrho[g];
    double vs = vsigma[g];
    double gr_x = grad_rho[g*3+0];
    double gr_y = grad_rho[g*3+1];
    double gr_z = grad_rho[g*3+2];

    int pos = g * nao + i;
    double phi = ao[pos];
    double dphi_x = gx[pos];
    double dphi_y = gy[pos];
    double dphi_z = gz[pos];

    double dot = gr_x * dphi_x + gr_y * dphi_y + gr_z * dphi_z;

    // Vxc Term = A^T * Y + Y^T * A
    // Decomposed into symmetric parts
    double term = 0.5 * vr * phi + 2.0 * vs * dot;
    
    Y_matrix[idx] = wg * term;
}

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

/* ---------- Host Functions ---------- */

extern "C" {

// OPTIMIZED: Uses cuBLAS GEMM instead of atomicAdd
void build_vxc_matrix_gga(int nao, int ngrid, 
                          const double *ao, 
                          const double *ao_grad, 
                          const double *weights,
                          const double *rho, const double *sigma, const double *grad_rho, 
                          double *vxc_mat) 
{
    size_t free_byte, total_byte; 
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE = size_t(free_byte * 0.9); 
    const size_t aux = 64 << 20;

    // Memory per row calculation:
    // AO (nao) + Grad (3*nao) + Y_temp (nao) + Scalars (7 doubles)
    const size_t per_row = (nao * 5 + 7) * sizeof(double); 
    size_t rows_per = (SAFE > aux) ? (SAFE - aux) / per_row : 0; 
    if(rows_per == 0) exit(1); 
    if(rows_per > ngrid) rows_per = ngrid;

    double *d_ao=0, *d_grad=0, *d_w=0, *d_rho=0, *d_sig=0, *d_grho=0, *d_vr=0, *d_vs=0, *d_mat=0;
    double *d_temp_Y = 0;

    CUDA_CHECK(cudaMalloc(&d_ao, rows_per*nao*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per*3*nao*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_temp_Y, rows_per*nao*sizeof(double))); // New buffer for Y

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
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

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

        const double *src_x = ao_grad + (size_t)0 * ngrid * nao + (size_t)g0 * nao;
        const double *src_y = ao_grad + (size_t)1 * ngrid * nao + (size_t)g0 * nao;
        const double *src_z = ao_grad + (size_t)2 * ngrid * nao + (size_t)g0 * nao;

        CUDA_CHECK(cudaMemcpyAsync(d_gx, src_x, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gy, src_y, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gz, src_z, copy_size_ao, cudaMemcpyHostToDevice));

        // 1. Compute Vrho/Vsigma
        int grid = (rows + BLOCK - 1) / BLOCK;
        gga_exc_vxc_kernel<<<grid, BLOCK>>>(rows, d_rho, d_sig, nullptr, d_vr, d_vs);
        
        // 2. Prepare Y Matrix
        int total_threads = rows * nao;
        int grid2 = (total_threads + BLOCK - 1) / BLOCK;
        prepare_gga_potentials_kernel<<<grid2, BLOCK>>>(
            rows, nao, d_ao, d_gx, d_gy, d_gz, 
            d_w, d_vr, d_vs, d_grho, d_temp_Y
        );

        // 3. GEMM Accumulation: V += A^T * Y + Y^T * A
        // cuBLAS handles ColMajor. RowMajor A(rows,nao) looks like A^T(nao,rows) to cuBLAS.
        // We want C_row = A_row^T * Y_row.
        // In ColMajor logic this is C_col = Y_col^T * A_col (where A_col = A_row^T).
        // This effectively computes V_ij = sum_k Y[k,i] * A[k,j].
        // We perform two GEMMs to sum both symmetric parts.

        double alpha = 1.0;
        double beta = 1.0; // Accumulate

        // Part 1: V += Y^T * A (Symmetric part 1)
        // d_temp_Y (A in GEMM), d_ao (B in GEMM) -> OP_N, OP_T
        // Result is (nao x nao)
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 nao, nao, rows,
                                 &alpha,
                                 d_temp_Y, nao,  // "A" (ColMajor N x K) -> Logic Y^T
                                 d_ao,     nao,  // "B" (ColMajor N x K) -> Logic A^T
                                 &beta,
                                 d_mat,    nao)); // C

        // Part 2: V += A^T * Y (Symmetric part 2)
        // Swap inputs
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 nao, nao, rows,
                                 &alpha,
                                 d_ao,     nao,
                                 d_temp_Y, nao,
                                 &beta,
                                 d_mat,    nao));
    }

    CUDA_CHECK(cudaMemcpy(vxc_mat, d_mat, (size_t)nao*nao*sizeof(double), cudaMemcpyDeviceToHost));
    
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad)); CUDA_CHECK(cudaFree(d_temp_Y));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); 
    CUDA_CHECK(cudaFree(d_grho)); CUDA_CHECK(cudaFree(d_vr)); CUDA_CHECK(cudaFree(d_vs)); 
    CUDA_CHECK(cudaFree(d_mat));
}

// OPTIMIZED: Uses cuBLAS Ddot for Exc summation
double compute_exc_energy_gga(int ngrid, const double *weights, const double *rho, const double *sigma) {
    size_t free_byte,total_byte; CUDA_CHECK(cudaMemGetInfo(&free_byte,&total_byte));
    const size_t SAFE=size_t(free_byte*0.9); const size_t aux=64<<20;
    const size_t per_row=4*sizeof(double); size_t rows_per=(SAFE>aux)?(SAFE-aux)/per_row:0;
    if(rows_per==0) exit(1); if(rows_per>ngrid) rows_per=ngrid;

    double *d_w,*d_rho,*d_sig,*d_exc;
    CUDA_CHECK(cudaMalloc(&d_w, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_exc, rows_per*sizeof(double)));
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int BLOCK=256; 
    double exc_total=0.0;
    
    for(int g0=0;g0<ngrid;g0+=rows_per){
        int g1=std::min(g0+(int)rows_per,ngrid); int rows=g1-g0;
        
        CUDA_CHECK(cudaMemcpyAsync(d_w, weights+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho, rho+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig, sigma+g0, rows*sizeof(double),cudaMemcpyHostToDevice));
        
        int grid=(rows+BLOCK-1)/BLOCK;
        gga_exc_vxc_kernel<<<grid,BLOCK>>>(rows,d_rho,d_sig,d_exc,nullptr,nullptr);
        
        double partial = 0.0;
        // Dot product: sum(w[i] * exc[i])
        CUBLAS_CHECK(cublasDdot(handle, rows, d_w, 1, d_exc, 1, &partial));
        exc_total += partial;
    }
    
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig));
    CUDA_CHECK(cudaFree(d_exc)); 
    return exc_total;
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

} // extern "C"

extern "C" void solve_fock_eigen(int n, const double *F_in, const double *S_in, double *e, double *C) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> F(n,n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> S(n,n);
    std::memcpy(F.data(), F_in, n*n*sizeof(double));
    std::memcpy(S.data(), S_in, n*n*sizeof(double));
    GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> solver(F, S, Eigen::ComputeEigenvectors);
    VectorXd evalues = solver.eigenvalues();
    auto evecs = solver.eigenvectors();
    std::memcpy(e, evalues.data(),  n*sizeof(double));
    std::memcpy(C, evecs.data(),  n*n*sizeof(double));
}