/* dft_cuda.cpp  -- CUDA-accelerated version with pointer-safety and double-atomic fallback */
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cuda_runtime.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::RowMajor;

/* ---------- error check macro ---------- */
#define CUDA_CHECK(call)                                             \
do {                                                                 \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << err << " '" << cudaGetErrorString(err) << "'" << std::endl; \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
} while(0)


/* ---------- 1. LDA VWN 参数 ---------- */
struct VWNPar {
    double A, b, c, x0;
};
static const VWNPar vwn_param_host[2] = {
    {0.0310907,  3.72744, 12.9352, -0.10498},   // ζ=0
    {0.01554535, 7.06042, 18.0578, -0.32500}    // ζ=1
};

/* For device use, copy parameters to constant memory */
__constant__ VWNPar vwn_param[2];

/* ---------- device double atomicAdd fallback ---------- */
__device__ inline double atomicAdd_double(double *address, double val) {
#if __CUDA_ARCH__ >= 600
    // On modern architectures, use hardware atomicAdd for double
    return atomicAdd(address, val);
#else
    // Fallback implementation using atomicCAS on 64-bit integer representation
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    double old_val;
    do {
        assumed = old;
        old_val = __longlong_as_double(assumed);
        unsigned long long int new_val_ull = __double_as_longlong(old_val + val);
        old = atomicCAS(address_as_ull, assumed, new_val_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

/* ---------- 1b. device math helpers ---------- */
__device__ inline void vwn_ec_device(double x, const VWNPar &p, double &ec, double &dec_dx)
{
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
    dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                    p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
}

/* ---------- 1c. LDA kernel: compute exc and vxc per grid point ---------- */
__global__ void lda_exc_vxc_kernel(int ngrid, const double *rho, double *exc, double *vxc, double zeta)
{
    const double pi = 3.14159265358979323846;
    const double Cx = 0.7385587663820224;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;

    double r = rho[g];
    if (r < 1e-300) r = 1e-300;
    double rs = pow(3.0 / (4.0 * pi * r), 1.0 / 3.0);
    double x  = sqrt(rs);

    // correlation
    double ec0 = 0.0, dec0_dx = 0.0, ec1 = 0.0, dec1_dx = 0.0;
    vwn_ec_device(x, vwn_param[0], ec0, dec0_dx);
    vwn_ec_device(x, vwn_param[1], ec1, dec1_dx);

    double z2 = zeta * zeta;
    double ec     = ec0 + (ec1 - ec0) * z2;
    double dec_dx = dec0_dx + (dec1_dx - dec0_dx) * z2;
    double vc     = ec - rs / 3.0 * dec_dx / (2.0 * x);

    // exchange
    double rho13 = pow(r, 1.0 / 3.0);
    double ex    = -Cx * r * rho13;
    double vx    = -4.0 / 3.0 * Cx * rho13;

    if (exc) exc[g] = ex + r * ec;
    if (vxc) vxc[g] = vx + vc;
}

/* ---------- 2. build_vxc_matrix on GPU ---------- */
/*
  Kernel computes each matrix element (i,j) by summing over grids:
    vxc_mat[i,j] = sum_g weights[g] * vxc[g] * ao[g*nao + i] * ao[g*nao + j]
*/
__global__ void build_vxc_matrix_kernel(int nao, int rows,
                                        int g0,
                                        const double *ao_b,   // (rows, nao)
                                        const double *w_b,    // (rows)
                                        const double *vxc_b,  // (rows)
                                        double *vxc_mat)      // 全局 (nao, nao)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nao) return;
    int im = idx / nao;        // 0..rows-1
    int i  = idx % nao;
    int g  = g0 + im;          // 全局网格指标

    double aoi = ao_b[im * nao + i];
    double w   = w_b[im];
    double vxc = vxc_b[im];

    /* 对 j 循环展开，原子加 */
    for (int j = 0; j < nao; ++j) {
        double aoj = ao_b[im * nao + j];
        double contrib = w * vxc * aoi * aoj;
        atomicAdd_double(&vxc_mat[i * nao + j], contrib);
    }
}


/* ---------- 3. compute_exc_energy on GPU (uses lda kernel then reduction) ---------- */
__global__ void weighted_sum_kernel(const double *weights, const double *values, double *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    double tmp = weights[tid] * values[tid];
    atomicAdd_double(out, tmp);
}

/* ---------- 4. get_rho on GPU ---------- */
__global__ void get_rho_kernel(int nao, 
                            int ngrid, 
                            const double *dm, 
                            const double *ao, double *rho_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;
    const double *phi_g = ao + (size_t)g * nao;
    double r = 0.0;
    for (int u = 0; u < nao; ++u) {
        double phiu = phi_g[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) {
            r += dm_row[v] * phiu * phi_g[v];
        }
    }
    rho_out[g] = r;
}

/* ---------- 5. build_coulomb_matrix on GPU (naive direct contraction) ---------- */
__global__ void build_coulomb_kernel(int nao, int rows_m, int m0,
                                     const double *eri_slice, // (rows_m,nao,nao,nao)
                                     const double *dm,
                                     double *J)               // 全局 (nao,nao)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = rows_m * nao;
    if (idx >= tot) return;
    int im = idx / nao;        // 0..rows_m-1
    int n  = idx % nao;
    int m  = m0 + im;          // 全局 m 指标

    double sum = 0.0;
    for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
            size_t pos = ((size_t)im * nao + n) * nao * nao + (size_t)l * nao + s;
            sum += dm[l * nao + s] * eri_slice[pos];
        }
    }
    atomicAdd_double(&J[m * nao + n], sum);
}

/* ---------- Host wrappers (extern "C") ---------- */
extern "C" {

/* Copy VWN parameters to device once */
static void copy_vwn_params_to_device()
{
    CUDA_CHECK(cudaMemcpyToSymbol(vwn_param, vwn_param_host, sizeof(VWNPar)*2));
}

/* build_vxc_matrix: GPU version */
void build_vxc_matrix(int nao, int ngrid,
                      const double *ao,    
                      const double *weights,
                      const double *rho,
                      double *vxc_mat)     
{
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = free_byte * 0.9;        

    const size_t aux_buf   = 64 * 1024 * 1024;       
    const size_t per_row   = (nao + 3) * sizeof(double); 
    const size_t left_byte = (SAFE_FREE > aux_buf) ? (SAFE_FREE - aux_buf) : 0;
    if (left_byte == 0) {
        std::cerr << "Not enough GPU memory to tile build_vxc_matrix!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t block_rows = left_byte / per_row;
    if (block_rows < 1) block_rows = 1;
    if (block_rows > ngrid) block_rows = ngrid;

    double *d_ao_b   = nullptr;   
    double *d_w_b    = nullptr;   
    double *d_rho_b  = nullptr;   
    double *d_vxc_b  = nullptr;   
    double *d_vxc_mat = nullptr;  
    CUDA_CHECK(cudaMalloc(&d_ao_b,    block_rows * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_b,     block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho_b,   block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc_b,   block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc_mat, (size_t)nao * nao * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vxc_mat, 0, (size_t)nao * nao * sizeof(double)));

    copy_vwn_params_to_device();

    const int BLOCK = 256;
    for (int g0 = 0; g0 < ngrid; g0 += block_rows) {
        int g1 = std::min(g0 + (int)block_rows, ngrid);
        int rows = g1 - g0;

        CUDA_CHECK(cudaMemcpyAsync(d_ao_b,   ao   + (size_t)g0 * nao,
                                   rows * nao * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_w_b,    weights + g0,
                                   rows * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho_b,  rho     + g0,
                                   rows * sizeof(double), cudaMemcpyHostToDevice));

        int grid_g = (rows + BLOCK - 1) / BLOCK;
        lda_exc_vxc_kernel<<<grid_g, BLOCK>>>(rows, d_rho_b, nullptr, d_vxc_b, 0.0);
        CUDA_CHECK(cudaGetLastError());

        int N = rows * nao;
        int grid = (N + BLOCK - 1) / BLOCK;
        build_vxc_matrix_kernel<<<grid, BLOCK>>>(
                nao, rows, g0, d_ao_b, d_w_b, d_vxc_b, d_vxc_mat);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(vxc_mat, d_vxc_mat, (size_t)nao * nao * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ao_b));
    CUDA_CHECK(cudaFree(d_w_b));
    CUDA_CHECK(cudaFree(d_rho_b));
    CUDA_CHECK(cudaFree(d_vxc_b));
    CUDA_CHECK(cudaFree(d_vxc_mat));
}

/* compute_exc_energy: GPU accelerated */
double compute_exc_energy(int ngrid,
                          const double *weights,
                          const double *rho)
{
    /* ---------- 1. 查剩余显存 ---------- */
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = free_byte * 0.9;          // 留 10 % 安全垫

    /* ---------- 2. 算每块最多多少行 ---------- */
    const size_t aux_buf   = 64 * 1024 * 1024;         // 64 MB 其它缓冲
    const size_t per_row   = 3 * sizeof(double);       // rho + weights + exc
    const size_t left_byte = (SAFE_FREE > aux_buf) ? (SAFE_FREE - aux_buf) : 0;
    if (left_byte == 0) {
        std::cerr << "Not enough GPU memory to tile compute_exc_energy!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t block_rows = left_byte / per_row;
    if (block_rows < 1) block_rows = 1;
    if (block_rows > ngrid) block_rows = ngrid;

    // std::cout << "[compute_exc_energy_tiled]  ngrid=" << ngrid
    //           << "  free=" << free_byte/1024/1024 << " MB"
    //           << "  block_rows=" << block_rows << std::endl;

    /* ---------- 3. 分配 device 内存（复用缓冲） ---------- */
    double *d_rho_b   = nullptr;
    double *d_w_b     = nullptr;
    double *d_exc_b   = nullptr;
    double *d_sum_b   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho_b,   block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_b,     block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc_b,   block_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_b,   sizeof(double)));

    /* ---------- 4. 把 VWN 参数拷进 constant memory ---------- */
    copy_vwn_params_to_device();

    /* ---------- 5. 分块计算 ---------- */
    const int BLOCK = 256;
    double exc_total = 0.0;
    for (int g0 = 0; g0 < ngrid; g0 += block_rows) {
        int g1 = std::min(g0 + (int)block_rows, ngrid);
        int rows = g1 - g0;

        /* 5a. H2D 拷当前子块 */
        CUDA_CHECK(cudaMemcpyAsync(d_rho_b, rho + g0,     rows * sizeof(double),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_w_b,   weights + g0, rows * sizeof(double),
                                   cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        lda_exc_vxc_kernel<<<grid, BLOCK>>>(rows, d_rho_b, d_exc_b, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_sum_b, 0, sizeof(double)));
        weighted_sum_kernel<<<grid, BLOCK>>>(d_w_b, d_exc_b, d_sum_b, rows);
        CUDA_CHECK(cudaGetLastError());

        double partial = 0.0;
        CUDA_CHECK(cudaMemcpy(&partial, d_sum_b, sizeof(double), cudaMemcpyDeviceToHost));
        exc_total += partial;
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---------- 6. 清理 ---------- */
    CUDA_CHECK(cudaFree(d_rho_b));
    CUDA_CHECK(cudaFree(d_w_b));
    CUDA_CHECK(cudaFree(d_exc_b));
    CUDA_CHECK(cudaFree(d_sum_b));

    return exc_total;
}

/* build_coulomb_matrix: GPU version (naive) */
void build_coulomb_matrix(int nao,
                          const double *eri,   // host pointer flattened 4-index
                          const double *dm,    // host pointer (nao,nao)
                          double *J)           // host pointer (nao,nao)
{
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = free_byte * 0.9;   // 留 10 % 安全垫

    const size_t dm_bytes   = (size_t)nao * nao * sizeof(double);
    const size_t j_bytes    = (size_t)nao * nao * sizeof(double);
    const size_t aux_buf    = 128 * 1024 * 1024;  // 128 MB 其它缓冲
    const size_t eri_row3   = (size_t)nao * nao * nao * sizeof(double); // 一个 m 切片
    const size_t left_bytes = (SAFE_FREE > dm_bytes + j_bytes + aux_buf) ?
                              (SAFE_FREE - dm_bytes - j_bytes - aux_buf) : 0;
    if (left_bytes == 0) {
        std::cerr << "Not enough GPU memory to tile build_coulomb!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t block_m = left_bytes / eri_row3;
    if (block_m < 1) block_m = 1;
    if (block_m > nao) block_m = nao;


    double *d_dm = nullptr;
    double *d_J  = nullptr;
    double *d_eri_slice = nullptr;  // 只存 [block_m][nao][nao][nao]
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes));
    CUDA_CHECK(cudaMalloc(&d_J,  j_bytes));
    CUDA_CHECK(cudaMalloc(&d_eri_slice, block_m * eri_row3));

    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J, 0, j_bytes));

    /* ---------- 4. 分块计算 ---------- */
    const int BLOCK = 256;
    for (int m0 = 0; m0 < nao; m0 += block_m) {
        int m1 = std::min(m0 + (int)block_m, nao);
        int rows = m1 - m0;  // 本次 m 方向行数

        /* 4a. H2D 拷 eri 切片: m 方向 [m0:m1] 全部 n,l,s */
        CUDA_CHECK(cudaMemcpyAsync(d_eri_slice,
                                   eri + (size_t)m0 * nao * nao * nao,
                                   (size_t)rows * eri_row3,
                                   cudaMemcpyHostToDevice));

        /* 4b. 对当前切片调用原内核，但把 “nao” 当成 “rows” 传进去，
              同时把全局偏移 m0 传进内核，让线程算出绝对 m 指标 */
        int grid = ((rows * nao) + BLOCK - 1) / BLOCK;
        build_coulomb_kernel<<<grid, BLOCK>>>(
                nao, rows, m0, d_eri_slice, d_dm, d_J);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---------- 5. 拷回结果 & 清理 ---------- */
    CUDA_CHECK(cudaMemcpy(J, d_J, j_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dm));
    CUDA_CHECK(cudaFree(d_J));
    CUDA_CHECK(cudaFree(d_eri_slice));
}

/* solve_fock_eigen: keep using Eigen on CPU */
void solve_fock_eigen(int n,
                      const double *F_in,
                      const double *S_in,
                      double *e,
                      double *C)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> F(n,n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> S(n,n);

    std::memcpy(F.data(), F_in, n*n*sizeof(double));
    std::memcpy(S.data(), S_in, n*n*sizeof(double));

    GeneralizedSelfAdjointEigenSolver<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor>
    > solver(F, S, Eigen::ComputeEigenvectors);

    VectorXd evalues = solver.eigenvalues();
    auto evecs = solver.eigenvectors();

    std::memcpy(e, evalues.data(), n*sizeof(double));
    std::memcpy(C, evecs.data(),  n*n*sizeof(double));
}

// void GPU_compute(int nao, int ngrid,
//              const double *dm,
//              const double *ao,
//              double *rho_out)
// {
    
// }

/* get_rho: GPU-accelerated version */
void get_rho(int nao, int ngrid,
             const double *dm,
             const double *ao,
             double *rho_out)
{
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = free_byte * 0.9;    

    const size_t dm_bytes   = (size_t)nao * nao * sizeof(double);  
    const size_t row_bytes  = (size_t)nao * sizeof(double);      
    const size_t rho_bytes  = sizeof(double);                    
    const size_t aux_buf    = 64 * 1024 * 1024;                  
    const size_t left_bytes = (SAFE_FREE > dm_bytes + aux_buf) ?
                              (SAFE_FREE - dm_bytes - aux_buf) : 0;
    if (left_bytes == 0) {
        std::cerr << "Not enough GPU memory to tile get_rho!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t rows_per_block = left_bytes / (row_bytes + rho_bytes);
    if (rows_per_block < 1) rows_per_block = 1;
    if (rows_per_block > ngrid) rows_per_block = ngrid;



    double *d_dm = nullptr;
    double *d_ao_block = nullptr;   
    double *d_rho_block = nullptr;  
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes));
    CUDA_CHECK(cudaMalloc(&d_ao_block, rows_per_block * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho_block, rows_per_block * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 128;    
    for (int g0 = 0; g0 < ngrid; g0 += rows_per_block) {
        int g1 = std::min(g0 + (int)rows_per_block, ngrid);
        int rows = g1 - g0;

        CUDA_CHECK(cudaMemcpyAsync(d_ao_block,
                                   ao + (size_t)g0 * nao,
                                   (size_t)rows * nao * sizeof(double),
                                   cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        get_rho_kernel<<<grid, BLOCK>>>(nao, rows, d_dm, d_ao_block, d_rho_block);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(rho_out + g0,
                                   d_rho_block,
                                   rows * sizeof(double),
                                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());  
    }


    CUDA_CHECK(cudaFree(d_dm));
    CUDA_CHECK(cudaFree(d_ao_block));
    CUDA_CHECK(cudaFree(d_rho_block));
}

} 
/* ---------- end of file ---------- */