/*
 * gga.cu – PBE-GGA (Math Corrected Ver.)
 * 1. Corrected t^2 scaling factor (Pi/16).
 * 2. Added missing dA/drho chain rule term in v_rho.
 * 3. Removed extra factor 2.0 in v_sigma (handled in kernel).
 * 4. Symmetrized matrix assembly.
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

#define RHO_EPS 1e-14 
#define MIN_GRAD 1e-24

// PW92 Constants
__constant__ double A_pw92 = 0.03109069086965489503; 
__constant__ double alpha1 = 0.21370;
__constant__ double beta1  = 7.5957;
__constant__ double beta2  = 3.5876;
__constant__ double beta3  = 1.6382;
__constant__ double beta4  = 0.49294;

/* ---------- Device Functions ---------- */

__device__ inline void pw92_correlation_rks(double rho, double &ec, double &vc) {
    if (rho <= RHO_EPS) { ec = 0.0; vc = 0.0; return; }
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

__device__ inline void pbe_exchange(double rho,double sigma, double &ex,double &vrho,double &vsigma){
    if(rho < RHO_EPS) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
    const double Cx = -0.7385587663820224; 
    const double kappa = 0.804;
    const double mu = 0.2195149727645171; 
    double rho13 = pow(rho, 1.0/3.0);
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    double s2 = (sigma > MIN_GRAD) ? sigma / (4.0 * kF * kF * rho * rho) : 0.0;
    if(s2 > 1e12) s2 = 1e12; 
    double denom = 1.0 + mu * s2 / kappa;
    double F = 1.0 + kappa - kappa / denom;
    ex = Cx * rho13 * F; 
    double dF_ds2 = mu / (denom * denom);
    vsigma = Cx * rho * rho13 * dF_ds2 * (1.0 / (4.0 * kF * kF * rho * rho));
    vrho = (4.0/3.0) * ex - (8.0/3.0) * Cx * rho13 * s2 * dF_ds2;
}

__device__ inline void pbe_correlation(double rho,double sigma, double &ec,double &vrho,double &vsigma){
    if(rho < RHO_EPS) { ec=0.0; vrho=0.0; vsigma=0.0; return; }
    double ec_lda, vc_lda;
    pw92_correlation_rks(rho, ec_lda, vc_lda);
    const double beta = 0.066725;
    const double gamma = 0.03109069086965489503; 
    double kF = pow(3.0*M_PI*M_PI*rho, 1.0/3.0);
    
    // --- FIX 1: Correct t^2 scaling (Pi/16) ---
    // t^2 = sigma * pi / (16 * kF * rho^2)
    double t2 = 0.0;
    if(sigma > MIN_GRAD) {
        double denom_t2 = 16.0 * kF * rho * rho;
        if(denom_t2 > 1e-100) t2 = sigma * M_PI / denom_t2; // Multiplied by PI
        else t2 = 1e20;
    }
    if(t2 > 1e20) t2 = 1e20;

    double x = -ec_lda / gamma;
    if(x > 100.0) x = 100.0; 
    double expm1_val = expm1(x); 
    double A_val = 0.0;
    if (fabs(expm1_val) < 1e-20) A_val = 1e20;
    else A_val = (beta / gamma) / expm1_val;

    double At2 = A_val * t2;
    double num_Q = 1.0 + At2;
    double den_Q = 1.0 + At2 + At2*At2;
    double Q = num_Q / den_Q;
    double term_in_log = 1.0 + (beta/gamma) * t2 * Q;
    double H = gamma * log(term_in_log);
    ec = ec_lda + H;

    // Derivatives
    double pre_log = gamma / term_in_log * (beta/gamma); 
    double Q_prime_num = -At2*At2 - 2.0*At2;
    double Q_prime = Q_prime_num / (den_Q * den_Q); 
    double d_term_dt2 = Q + At2 * Q_prime; 
    double dH_dt2 = pre_log * d_term_dt2;
    double d_term_dA = t2 * t2 * Q_prime;
    double dH_dA = pre_log * d_term_dA;

    // --- FIX 2: Correct vsigma derivative (removed factor 2.0) ---
    // dt^2/dsigma = pi / (16 * kF * rho^2)
    double dt2_dsigma = M_PI / (8.0 * kF * rho * rho);
    vsigma = rho * dH_dt2 * dt2_dsigma; 
    
    const double t2_safe   = fmax(t2, 1.0e-16);
    const double kF_rho2   = 16.0 * kF * rho * rho;
    const double dt2_dsig  = M_PI / fmax(kF_rho2, 1.0e-60);   // 8 已体现在分子 PI

    vsigma = rho * dH_dt2 * dt2_dsig;                         // 相关势对 |∇n|
    vsigma = fmin(fmax(vsigma, -1.0e3), 1.0e3);
    /* ---------- vrho 链 ---------- */
    double exp_val     = exp(x);
    double dA_dec      = A_val * exp_val / (expm1_val * gamma);
    double rho_dA_drho = dA_dec * (vc_lda - ec_lda);          // chain rule
    double rho_dt2_drho = -7.0 / 3.0 * t2_safe;               // 用 safe t2
    double rho_dH_drho = dH_dA * rho_dA_drho  +  dH_dt2 * rho_dt2_drho;

    vrho = vc_lda + H + rho_dH_drho;
 
}

/* ---------- Kernels ---------- */

__global__ void gga_exc_vxc_kernel(int ngrid, const double *rho, const double *sigma, double *exc, double *vrho, double *vsigma){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=ngrid) return;
    double r_val=rho[g];
    double s_val=sigma[g];
    if(r_val < RHO_EPS) {
        if(exc) exc[g] = 0.0;
        if(vrho) vrho[g] = 0.0;
        if(vsigma) vsigma[g] = 0.0;
        return;
    }
    double ex, vrx, vsx;
    double ec, vrc, vsc;
    pbe_exchange(r_val, s_val, ex, vrx, vsx);

    pbe_correlation(r_val, s_val, ec, vrc, vsc);
    if(exc)    exc[g]    = r_val * (ex + ec); 
    if(vrho)   vrho[g]   = vrx + vrc;
    if(vsigma) vsigma[g] = vsx + vsc;

}


__global__ void get_rho_sigma_kernel_planar(int nao, int rows,
                                            const double *dm, 
                                            const double *ao, 
                                            const double *gx, // 指向本 Batch 的 X 梯度
                                            const double *gy, // 指向本 Batch 的 Y 梯度
                                            const double *gz, // 指向本 Batch 的 Z 梯度
                                            double *rho, double *sigma, double *grad_rho_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows) return;

    // 1. 读取 AO 数值
    const double *phi = ao + (size_t)g * nao;
    
    // 2. 读取 梯度 (现在是 Planar 模式，内存直接连续)
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
            
            // 计算密度
            r += dm_val * phiu * phiv;
            
            // 计算梯度: d(uv) = u'v + uv'
            double term_x = dx_u * phiv + phiu * gphi_x[v];
            double term_y = dy_u * phiv + phiu * gphi_y[v];
            double term_z = dz_u * phiv + phiu * gphi_z[v];

            gr_x += dm_val * term_x;
            gr_y += dm_val * term_y;
            gr_z += dm_val * term_z;
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
// Assumes ao_grad is Interleaved [ngrid][3][nao]
__global__ void get_rho_sigma_kernel(int nao, int rows, const double *dm, const double *ao, const double *ao_grad,
                                     double *rho, double *sigma, double *grad_rho_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows) return;
    const double *phi_row = ao + (size_t)g * nao;
    double r = 0.0;
    for (int u = 0; u < nao; ++u) {
        double phiu = phi_row[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) r += dm_row[v] * phiu * phi_row[v];
    }
    rho[g] = r;
    double grd[3] = {0.0, 0.0, 0.0};
    const double *gphi = ao_grad + (size_t)g * 3 * nao; 
    for (int d = 0; d < 3; ++d) {
        const double *gphi_d = gphi + d * nao;
        for (int u = 0; u < nao; ++u) {
            double phi_u = phi_row[u];
            double gphi_du = gphi_d[u];
            const double *dm_row = dm + (size_t)u * nao;
            for (int v = 0; v < nao; ++v)
                grd[d] += dm_row[v] * (gphi_du * phi_row[v] + phi_u * gphi_d[v]);
        }
    }
    sigma[g] = grd[0]*grd[0] + grd[1]*grd[1] + grd[2]*grd[2];
    if(grad_rho_out){ grad_rho_out[g*3+0] = grd[0]; grad_rho_out[g*3+1] = grd[1]; grad_rho_out[g*3+2] = grd[2]; }
}

// 修改后的 Vxc Matrix Kernel (Planar 模式)
// 注意：移除了 ao_grad_b 指针，改为 gx_b, gy_b, gz_b
__global__ void build_vxc_matrix_gga_kernel_planar(
    int nao, int rows, int g0,
    const double *ao_b,
    const double *gx_b, // X 梯度
    const double *gy_b, // Y 梯度
    const double *gz_b, // Z 梯度
    const double *w_b,
    const double *vrho_b,
    const double *vsigma_b,
    const double *grad_rho_b, 
    double *vxc_mat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx 覆盖 (Rows * NAO)
    if(idx >= rows * nao) return;
    
    int im = idx / nao; // 当前 Grid 在 batch 中的索引 (0..rows-1)
    int i  = idx % nao; // 当前 AO 索引

    double w = w_b[im];
    if(fabs(w) < 1e-15) return;

    // 读取当前 Grid 点的势函数值
    double vr = vrho_b[im];
    double vs = vsigma_b[im];
    
    // 读取当前 Grid 点的密度梯度 (grad_rho)
    // 假设 grad_rho 依然是 Interleaved [x,y,z] 或者是 Planar?
    // 通常 grad_rho 比较小，我们在 get_rho_sigma 里输出的是 interleaved [g*3+d]
    // 检查一下 get_rho_sigma_kernel 的输出，如果是 [g*3+0/1/2]，这里就不用改
    double gr_x = grad_rho_b[im*3+0];
    double gr_y = grad_rho_b[im*3+1];
    double gr_z = grad_rho_b[im*3+2];

    double aoi = ao_b[im*nao+i];
    
    // 读取 phi_i 的梯度 (Planar 读取)
    double dxi = gx_b[im*nao+i];
    double dyi = gy_b[im*nao+i];
    double dzi = gz_b[im*nao+i];

    // 计算 dot(nabla rho, nabla phi_i)
    double dot_rho_phi_i = gr_x * dxi + gr_y * dyi + gr_z * dzi;

    // 内层循环：遍历 j
    // 为了利用对称性并减少重复计算，建议 j 从 0 到 nao
    // (如果之前优化过 j<=i，这里保持一致即可，为了安全先用全循环)
    for(int j=0; j<nao; ++j){
        double aoj = ao_b[im*nao+j];
        
        // 读取 phi_j 的梯度
        double dxj = gx_b[im*nao+j];
        double dyj = gy_b[im*nao+j];
        double dzj = gz_b[im*nao+j];

        double dot_rho_phi_j = gr_x * dxj + gr_y * dyj + gr_z * dzj;

        // GGA Vxc 矩阵公式
        double term_rho = vr * aoi * aoj;
        double term_sig = 2.0 * vs * (aoi * dot_rho_phi_j + aoj * dot_rho_phi_i);
        
        // 原子累加
        atomicAdd(vxc_mat + i*nao+j, w * (term_rho + term_sig));
    }
}

// Symmetrized Matrix Build
__global__ void build_vxc_matrix_gga_kernel(int nao,int rows,int g0,
                                            const double *ao_b, const double *ao_grad_b,
                                            const double *w_b, const double *vrho_b, const double *vsigma_b,
                                            const double *grad_rho_b, double *vxc_mat){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=rows*nao) return;
    int im=idx/nao;
    int i =idx%nao;

    double w=w_b[im];
    if(abs(w) < 1e-15) return;

    double aoi=ao_b[im*nao+i];
    double vr=vrho_b[im], vs=vsigma_b[im];
    double gr[3];
    gr[0] = grad_rho_b[im*3+0]; gr[1] = grad_rho_b[im*3+1]; gr[2] = grad_rho_b[im*3+2];

    const double *gbase = ao_grad_b + (size_t)im * 3 * nao;
    double dot_rho_phi_i = 0.0;
    for(int d=0; d<3; ++d) dot_rho_phi_i += gr[d] * gbase[d*nao+i];

    // Symmetrized loop: j >= i
    for(int j=i; j<nao; ++j){
        double aoj=ao_b[im*nao+j];
        double dot_rho_phi_j = 0.0;
        for(int d=0; d<3; ++d) dot_rho_phi_j += gr[d] * gbase[d*nao+j];

        double term_rho = vr * aoi * aoj;
        double term_sig = 2.0 * vs * (aoi * dot_rho_phi_j + aoj * dot_rho_phi_i);
        double val = w * (term_rho + term_sig);

        atomicAdd_double(&vxc_mat[i*nao+j], val);
        if(i != j) {
            atomicAdd_double(&vxc_mat[j*nao+i], val);
        }
    }
}

// ... Rest is same as before ...
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
__global__ void weighted_sum_gga_kernel(const double *w,const double *exc,double *out,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    atomicAdd_double(out,w[tid]*exc[tid]);
}

extern "C" {

void build_vxc_matrix_gga(int nao, int ngrid, 
                                     const double *ao, 
                                     const double *ao_grad, // 假设输入 PySCF Planar (3, N, NAO)
                                     const double *weights,
                                     const double *rho, const double *sigma, const double *grad_rho, 
                                     double *vxc_mat) 
{
    size_t free_byte, total_byte; 
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE = size_t(free_byte * 0.9); 
    const size_t aux = 64 << 20;
    
    // per_row 计算：AO(1) + Grad(3) + W(1) + Rho(1) + Sig(1) + GRho(3) + Vr(1) + Vs(1)
    // Grad 现在虽然分三个指针，但总显存占用不变
    const size_t per_row = (nao * 4 + 7) * sizeof(double); 
    size_t rows_per = (SAFE > aux) ? (SAFE - aux) / per_row : 0; 
    if(rows_per == 0) exit(1); 
    if(rows_per > ngrid) rows_per = ngrid;

    double *d_ao=0, *d_grad=0, *d_w=0, *d_rho=0, *d_sig=0, *d_grho=0, *d_vr=0, *d_vs=0, *d_mat=0;
    
    // 显存分配
    CUDA_CHECK(cudaMalloc(&d_ao, rows_per*nao*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per*3*nao*sizeof(double))); // 依然申请一大块存梯度
    CUDA_CHECK(cudaMalloc(&d_w, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grho, rows_per*3*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_vr, rows_per*sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_vs, rows_per*sizeof(double)));
    
    // 矩阵初始化
    CUDA_CHECK(cudaMalloc(&d_mat, (size_t)nao*nao*sizeof(double))); 
    CUDA_CHECK(cudaMemset(d_mat, 0, (size_t)nao*nao*sizeof(double)));

    // 定义梯度指针别名
    double *d_gx = d_grad;
    double *d_gy = d_grad + rows_per * nao;
    double *d_gz = d_grad + 2 * rows_per * nao;

    const int BLOCK = 256;
    
    // --- 这里的 Kernel 调用需要先算 vrho/vsigma (gga_exc_vxc_kernel) --- 
    // 注意：gga_exc_vxc_kernel 只依赖 rho/sigma，不需要改动，它已经是正确的了
    
    for(int g0=0; g0<ngrid; g0+=rows_per){
        int g1 = std::min(g0+(int)rows_per, ngrid); 
        int rows = g1 - g0;
        size_t copy_size_ao = rows * nao * sizeof(double);

        // 1. 数据拷贝
        CUDA_CHECK(cudaMemcpyAsync(d_ao, ao + g0*nao, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_w, weights + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho, rho + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig, sigma + g0, rows*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_grho, grad_rho + g0*3, rows*3*sizeof(double), cudaMemcpyHostToDevice));

        // 2. 梯度拷贝 (Planar 模式)
        // src_x/y/z 计算逻辑同 get_rho_sigma
        const double *src_x = ao_grad + (size_t)0 * ngrid * nao + (size_t)g0 * nao;
        const double *src_y = ao_grad + (size_t)1 * ngrid * nao + (size_t)g0 * nao;
        const double *src_z = ao_grad + (size_t)2 * ngrid * nao + (size_t)g0 * nao;

        CUDA_CHECK(cudaMemcpyAsync(d_gx, src_x, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gy, src_y, copy_size_ao, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gz, src_z, copy_size_ao, cudaMemcpyHostToDevice));

        // 3. 计算 Vrho, Vsigma (这个 Kernel 不变)
        int grid = (rows + BLOCK - 1) / BLOCK;
        gga_exc_vxc_kernel<<<grid, BLOCK>>>(rows, d_rho, d_sig, nullptr, d_vr, d_vs);
        
        // 4. 计算 Vxc Matrix (调用新的 Planar Kernel)
        int N = rows * nao; 
        int grid2 = (N + BLOCK - 1) / BLOCK;
        
        build_vxc_matrix_gga_kernel_planar<<<grid2, BLOCK>>>(
            nao, rows, g0,
            d_ao, 
            d_gx, d_gy, d_gz, // 传入分开的梯度指针
            d_w, d_vr, d_vs, d_grho, 
            d_mat
        );
    }
    
    CUDA_CHECK(cudaMemcpy(vxc_mat, d_mat, (size_t)nao*nao*sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free ...
    CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_grho));
    CUDA_CHECK(cudaFree(d_vr)); CUDA_CHECK(cudaFree(d_vs)); CUDA_CHECK(cudaFree(d_mat));
}

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

void get_rho_sigma(int nao, int ngrid, 
                              const double *dm, 
                              const double *ao, 
                              const double *ao_grad, // 假设这里传入的是 PySCF 原生 (3, N, NAO) 数组的指针
                              double *rho, double *sigma, double *grad_rho) 
{
    size_t free_byte, total_byte; 
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE = size_t(free_byte * 0.9); 
    const size_t dm_bytes = nao * nao * sizeof(double);
    
    // GPU 端依然分配足够存所有梯度的空间
    const size_t row_bytes = (4 * nao + 5) * sizeof(double); 
    const size_t aux = 64 << 20;
    
    size_t rows_per = (SAFE > dm_bytes + aux) ? (SAFE - dm_bytes - aux) / row_bytes : 0;
    if (rows_per == 0) exit(1); 
    if (rows_per > ngrid) rows_per = ngrid;

    double *d_dm, *d_ao, *d_grad, *d_rho, *d_sig, *d_grho;
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes)); 
    CUDA_CHECK(cudaMalloc(&d_ao, rows_per * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per * 3 * nao * sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per * sizeof(double))); 
    CUDA_CHECK(cudaMalloc(&d_grho, rows_per * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));

    // 定义 d_grad 内部的三个分区指针（GPU显存内）
    double *d_gx = d_grad;
    double *d_gy = d_grad + rows_per * nao;
    double *d_gz = d_grad + 2 * rows_per * nao;

    const int BLOCK = 128;
    for (int g0 = 0; g0 < ngrid; g0 += rows_per) {
        int g1 = std::min(g0 + (int)rows_per, ngrid); 
        int rows = g1 - g0;
        size_t copy_size = rows * nao * sizeof(double);

        // 1. 拷贝 AO 数值
        CUDA_CHECK(cudaMemcpyAsync(d_ao, ao + g0 * nao, copy_size, cudaMemcpyHostToDevice));

        // 2. 关键修改：分别拷贝 X, Y, Z 分量
        // PySCF 内存布局： [X 全体] [Y 全体] [Z 全体]
        // X 起点: ao_grad + 0
        // Y 起点: ao_grad + 1 * ngrid * nao
        // Z 起点: ao_grad + 2 * ngrid * nao
        
        // 计算当前 Batch 在 Host 端源数据的偏移位置
        const double *src_x = ao_grad + (size_t)0 * ngrid * nao + (size_t)g0 * nao;
        const double *src_y = ao_grad + (size_t)1 * ngrid * nao + (size_t)g0 * nao;
        const double *src_z = ao_grad + (size_t)2 * ngrid * nao + (size_t)g0 * nao;

        CUDA_CHECK(cudaMemcpyAsync(d_gx, src_x, copy_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gy, src_y, copy_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_gz, src_z, copy_size, cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        
        // 调用新的 Kernel，传入分开的指针
        get_rho_sigma_kernel_planar<<<grid, BLOCK>>>(
            nao, rows, 
            d_dm, 
            d_ao, 
            d_gx, d_gy, d_gz, 
            d_rho, d_sig, d_grho
        );

        CUDA_CHECK(cudaMemcpyAsync(rho + g0, d_rho, rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(sigma + g0, d_sig, rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(grad_rho + g0 * 3, d_grho, rows * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
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