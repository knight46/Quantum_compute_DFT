#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error at line %d\n",__LINE__); } }while(0)

#define RHO_EPS 1e-20

// --- VWN Constants ---
struct VWNPar {
    double A, b, c, x0;
};

// VWN parameters for paramagnetic (zeta=0) and ferromagnetic (zeta=1)
// We store them in constant memory for speed
__constant__ VWNPar vwn_param[2];

// Host shadow for initialization
const VWNPar vwn_param_host[2] = {
    {0.0310907,  3.72744, 12.9352, -0.10498},   // Paramagnetic
    {0.01554535, 7.06042, 18.0578, -0.32500}    // Ferromagnetic
};

// --- Device Functions (Physics) ---

__device__ inline void slater_exchange(double rho, double &ex, double &vx) {
    if (rho < RHO_EPS) { ex = 0.0; vx = 0.0; return; }
    // Cx = (3/4) * (3/pi)^(1/3) approx 0.738558...
    const double Cx = 0.7385587663820224;
    double rho13 = pow(rho, 1.0/3.0);
    // Energy density per particle: epsilon_x = -Cx * rho^(1/3)
    // Note: PySCF convention treats 'ex' as energy density * rho in some contexts, 
    // but here we return epsilon_x (energy per particle).
    // The kernel will multiply by rho later.
    ex = -Cx * rho13;
    // Potential: v_x = -(4/3) * Cx * rho^(1/3)
    vx = (4.0/3.0) * ex;
}

__device__ inline void vwn_ec_sub(double x, const VWNPar &p, double &ec, double &dec_dx) {
    const double X = x * x + p.b * x + p.c;
    const double Q = sqrt(4.0 * p.c - p.b * p.b);
    const double log_term  = log(x * x / X);
    const double atan_term = 2.0 * p.b / Q * atan(Q / (2.0 * x + p.b));
    const double x02 = p.x0 * p.x0;
    const double denom = x02 + p.b * p.x0 + p.c;
    const double corr  = p.b * p.x0 / denom *
        (log((x - p.x0) * (x - p.x0) / X) +
         2.0 * (2.0 * p.x0 + p.b) / Q * atan(Q / (2.0 * x + p.b)));
    
    ec = p.A * (log_term + atan_term - corr);
    
    // d(ec)/dx
    dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                    p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
}

__device__ inline void vwn_correlation(double rho, double &ec, double &vc) {
    if (rho < RHO_EPS) { ec = 0.0; vc = 0.0; return; }
    
    const double pi = 3.14159265358979323846;
    double rs = pow(3.0 / (4.0 * pi * rho), 1.0 / 3.0);
    double x  = sqrt(rs);

    double ec0, dec0_dx; // Paramagnetic
    // Note: Since this is RKS, we theoretically only need ec0 (zeta=0).
    // However, to strictly follow standard VWN implementation structure:
    vwn_ec_sub(x, vwn_param[0], ec0, dec0_dx);
    
    // For RKS, ec = ec0
    ec = ec0;
    
    // vc = ec - (rs/3) * (dec/drs)
    // dx/drs = 1 / (2*sqrt(rs)) = 1 / (2x)
    // dec/drs = dec/dx * dx/drs = dec_dx / (2x)
    vc = ec - (rs / 3.0) * (dec0_dx / (2.0 * x));
}

// --- Kernels ---

// 1. Calculate Rho: rho[g] = sum_uv DM[uv] * phi_u[g] * phi_v[g]
// Optimized as: rho[g] = sum_u phi_u[g] * (sum_v DM[uv] * phi_v[g])
// But for simplicity/robustness matching the GGA one, we use a simple planar kernel first.
__global__ void get_rho_kernel(int nao, int ngrid, const double *dm, const double *ao, double *rho_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;

    const double *phi = ao + (size_t)g * nao; // Point to row g
    double val = 0.0;

    for (int u = 0; u < nao; ++u) {
        double phiu = phi[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) {
            val += dm_row[v] * phiu * phi[v];
        }
    }
    rho_out[g] = val;
}

// 2. Fused Kernel: Calculate Vxc parts or Exc parts
// If compute_B is true: populates B_mat for Vxc calculation
// If compute_B is false: populates exc_out for Energy calculation
__global__ void lda_fused_kernel(
    int ngrid, int nao, bool compute_B,
    const double *w, const double *rho, const double *ao,
    double *exc_out, double *B_mat
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Grid point index
    if (k >= ngrid) return;

    double r_val = rho[k];
    
    if (r_val < RHO_EPS) {
        if (!compute_B && exc_out) exc_out[k] = 0.0;
        if (compute_B) {
            for (int i = 0; i < nao; ++i) B_mat[k * nao + i] = 0.0;
        }
        return;
    }

    double ex=0, vx=0;
    double ec=0, vc=0;

    slater_exchange(r_val, ex, vx);
    vwn_correlation(r_val, ec, vc);

    // Compute Energy: Exc = integral( rho * (eps_x + eps_c) )
    if (!compute_B && exc_out) {
        exc_out[k] = r_val * (ex + ec); 
        return; 
    }

    // Compute Potential Intermediate B
    // V_xc_uv = sum_k w_k * (vx + vc) * phi_u[k] * phi_v[k]
    // We compute B[k, i] = w_k * (vx + vc) * phi_i[k]
    // Then V_xc = AO^T * B
    if (compute_B) {
        double v_total = vx + vc;
        double weight = w[k];
        double factor = weight * v_total;

        const double *phi_row = ao + k * nao;
        double *B_row = B_mat + k * nao;

        for (int i = 0; i < nao; ++i) {
            B_row[i] = factor * phi_row[i];
        }
    }
}

// 3. Reduction Kernel for Energy Summation
__global__ void reduce_sum_kernel(const double *w, const double *val, double *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += w[i] * val[i];
    }
    atomicAdd(out, sum);
}

// --- Extern C Interface ---

extern "C" {

static cublasHandle_t handle = nullptr;
static bool vwn_init = false;

// Helper to init constants once
void check_vwn_init() {
    if (!vwn_init) {
        cudaMemcpyToSymbol(vwn_param, vwn_param_host, sizeof(VWNPar)*2);
        vwn_init = true;
    }
}

void get_rho_gpu(int nao, int ngrid, const double *d_dm, const double *d_ao, double *d_rho) {
    int block = 128;
    int grid = (ngrid + block - 1) / block;
    get_rho_kernel<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_rho);
}

void build_vxc_gpu(int nao, int ngrid, 
                   const double *d_ao, const double *d_weights, const double *d_rho,
                   double *d_B_work, double *d_vxc) 
{
    if (!handle) cublasCreate(&handle);
    check_vwn_init();

    int block = 256;
    int grid = (ngrid + block - 1) / block;
    
    // 1. Calculate B matrix: B[g, i] = w[g] * vxc[g] * ao[g, i]
    lda_fused_kernel<<<grid, block>>>(
        ngrid, nao, true, 
        d_weights, d_rho, d_ao, 
        nullptr, d_B_work
    );

    // 2. Calculate Vxc = AO^T * B
    // AO is (ngrid, nao) in RowMajor (C-order).
    // cuBLAS interprets pointer as (nao, ngrid) ColMajor.
    // effectively, we have A_gpu = AO^T.
    // B_gpu = B^T.
    // We want V = AO^T * B.
    // In cuBLAS: C = A * B^T?? No.
    // Let's analyze gga logic:
    // We want sum_g ( AO[g,i] * B[g,j] ).
    // GPU_A is (nao x ngrid). GPU_B is (nao x ngrid).
    // C = GPU_A * GPU_B^T = (nao x ngrid) * (ngrid x nao) = (nao x nao).
    // This perfectly matches standard dense matmul logic on GPU for this layout.
    
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nao, nao, ngrid,
        &alpha,
        d_ao, nao,      // A (viewed as Transpose of Python array)
        d_B_work, nao,  // B (viewed as Transpose of Python array)
        &beta,
        d_vxc, nao      // C
    );
}

double compute_exc_gpu(int ngrid, int nao, const double *d_weights, 
                       const double *d_rho, double *d_exc_work)
{
    check_vwn_init();
    
    int block = 256;
    int grid = (ngrid + block - 1) / block;

    // 1. Compute Exc density per grid point
    lda_fused_kernel<<<grid, block>>>(
        ngrid, nao, false, 
        nullptr, d_rho, nullptr, 
        d_exc_work, nullptr
    );

    // 2. Integrate: sum(weight * exc_density)
    double *d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));
    reduce_sum_kernel<<<256, 256>>>(d_weights, d_exc_work, d_sum, ngrid);
    
    double val;
    cudaMemcpy(&val, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return val;
}

// Same logic as GGA for J matrix
void build_coulomb_gpu(int nao, const double *d_eri, const double *d_dm, double *d_J) {
    if (!handle) cublasCreate(&handle);

    // Treat ERI as matrix (N^2, N^2) and DM as vector (N^2)
    // J = ERI * DM
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