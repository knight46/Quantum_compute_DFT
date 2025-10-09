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
__global__ void build_vxc_matrix_kernel(int nao, int ngrid,
                                        const double *ao,     // (ngrid, nao)
                                        const double *weights,
                                        const double *vxc,    // (ngrid)
                                        double *vxc_mat)      // (nao,nao) row-major
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nao * nao;
    if (idx >= N) return;
    int i = idx / nao;
    int j = idx % nao;
    double sum = 0.0;
    for (int g = 0; g < ngrid; ++g) {
        double aoi = ao[(size_t)g * nao + i];
        double aoj = ao[(size_t)g * nao + j];
        sum += weights[g] * vxc[g] * aoi * aoj;
    }
    vxc_mat[i * nao + j] = sum;
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
__global__ void build_coulomb_kernel(int nao, const double *eri, const double *dm, double *J)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nao * nao;
    if (idx >= N) return;
    int m = idx / nao;
    int n = idx % nao;
    double sum = 0.0;
    for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
            size_t pos = ((size_t)m * nao + n) * nao * nao + (size_t)l * nao + s;
            sum += dm[(size_t)l * nao + s] * eri[pos];
        }
    }
    J[m * nao + n] = sum;
}

/* ---------- Host wrappers (extern "C") ---------- */
extern "C" {

/* Copy VWN parameters to device once */
static void copy_vwn_params_to_device()
{
    CUDA_CHECK(cudaMemcpyToSymbol(vwn_param, vwn_param_host, sizeof(VWNPar)*2));
}

/* lda_exc_vxc: host wrapper that runs GPU kernel */
void lda_exc_vxc(int n, const double *rho_host, double *exc_host, double *vxc_host, double zeta)
{
    copy_vwn_params_to_device();

    double *d_rho = nullptr;
    double *d_exc = nullptr;
    double *d_vxc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rho, rho_host, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;
    // We provide both exc and vxc buffers here.
    lda_exc_vxc_kernel<<<grid, block>>>(n, d_rho, d_exc, d_vxc, zeta);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(exc_host, d_exc, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vxc_host, d_vxc, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_exc));
    CUDA_CHECK(cudaFree(d_vxc));
}

/* build_vxc_matrix: GPU version */
void build_vxc_matrix(int nao, int ngrid,
                      const double *ao,     // host pointer (ngrid, nao)
                      const double *weights,
                      const double *rho,
                      double *vxc_mat)      // host pointer (nao, nao)
{
    // allocate device memory
    double *d_ao = nullptr;
    double *d_weights = nullptr;
    double *d_rho = nullptr;
    double *d_vxc = nullptr;
    double *d_vxc_mat = nullptr;

    size_t ao_bytes = (size_t)ngrid * nao * sizeof(double);
    size_t weights_bytes = (size_t)ngrid * sizeof(double);
    size_t vxc_bytes = (size_t)ngrid * sizeof(double);
    size_t mat_bytes = (size_t)nao * nao * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_ao, ao_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_rho, (size_t)ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, vxc_bytes));
    CUDA_CHECK(cudaMalloc(&d_vxc_mat, mat_bytes));

    CUDA_CHECK(cudaMemcpy(d_ao, ao, ao_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, rho, (size_t)ngrid * sizeof(double), cudaMemcpyHostToDevice));

    // compute vxc on device (we only need vxc here, but kernel is safe with both buffers present)
    copy_vwn_params_to_device();
    int block_g = 256;
    int grid_g = (ngrid + block_g - 1) / block_g;
    lda_exc_vxc_kernel<<<grid_g, block_g>>>(ngrid, d_rho, d_vxc /*exc not needed? but provide*/, d_vxc, 0.0);
    CUDA_CHECK(cudaGetLastError());

    // compute vxc_mat: each thread computes one (i,j) entry
    int N = nao * nao;
    int block = 256;
    int grid = (N + block - 1) / block;
    build_vxc_matrix_kernel<<<grid, block>>>(nao, ngrid, d_ao, d_weights, d_vxc, d_vxc_mat);
    CUDA_CHECK(cudaGetLastError());

    // copy back
    CUDA_CHECK(cudaMemcpy(vxc_mat, d_vxc_mat, mat_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ao));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_vxc));
    CUDA_CHECK(cudaFree(d_vxc_mat));
}

/* compute_exc_energy: GPU accelerated */
double compute_exc_energy(int ngrid,
                          const double *weights,
                          const double *rho)
{
    double *d_rho = nullptr;
    double *d_weights = nullptr;
    double *d_exc = nullptr;
    double *d_vxc = nullptr;
    double *d_sum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rho, (size_t)ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, (size_t)ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, (size_t)ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rho, rho, (size_t)ngrid * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, (size_t)ngrid * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    copy_vwn_params_to_device();
    int block_g = 256;
    int grid_g = (ngrid + block_g - 1) / block_g;
    lda_exc_vxc_kernel<<<grid_g, block_g>>>(ngrid, d_rho, d_exc, d_vxc, 0.0);
    CUDA_CHECK(cudaGetLastError());

    int block = 256;
    int grid = (ngrid + block - 1) / block;
    weighted_sum_kernel<<<grid, block>>>(d_weights, d_exc, d_sum, ngrid);
    CUDA_CHECK(cudaGetLastError());

    double exc_sum_host = 0.0;
    CUDA_CHECK(cudaMemcpy(&exc_sum_host, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_exc));
    CUDA_CHECK(cudaFree(d_vxc));
    CUDA_CHECK(cudaFree(d_sum));

    return exc_sum_host;
}

/* build_coulomb_matrix: GPU version (naive) */
void build_coulomb_matrix(int nao,
                          const double *eri,   // host pointer flattened 4-index
                          const double *dm,    // host pointer (nao,nao)
                          double *J)           // host pointer (nao,nao)
{
    size_t eri_size = (size_t)nao * nao * nao * nao * sizeof(double);
    size_t dm_size = (size_t)nao * nao * sizeof(double);
    size_t mat_size = (size_t)nao * nao * sizeof(double);

    double *d_eri = nullptr;
    double *d_dm = nullptr;
    double *d_J = nullptr;

    CUDA_CHECK(cudaMalloc(&d_eri, eri_size));
    CUDA_CHECK(cudaMalloc(&d_dm, dm_size));
    CUDA_CHECK(cudaMalloc(&d_J, mat_size));

    CUDA_CHECK(cudaMemcpy(d_eri, eri, eri_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J, 0, mat_size));

    int N = nao * nao;
    int block = 256;
    int grid = (N + block - 1) / block;
    build_coulomb_kernel<<<grid, block>>>(nao, d_eri, d_dm, d_J);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(J, d_J, mat_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_eri));
    CUDA_CHECK(cudaFree(d_dm));
    CUDA_CHECK(cudaFree(d_J));
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

/* get_rho: GPU-accelerated version */
void get_rho(int nao, int ngrid,
             const double *dm,
             const double *ao,
             double *rho_out)
{
    double *d_dm = nullptr;
    double *d_ao = nullptr;
    double *d_rho = nullptr;

    size_t dm_size = (size_t)nao * nao * sizeof(double);
    size_t ao_size = (size_t)ngrid * nao * sizeof(double);
    size_t rho_size = (size_t)ngrid * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_dm, dm_size));
    CUDA_CHECK(cudaMalloc(&d_ao, ao_size));
    CUDA_CHECK(cudaMalloc(&d_rho, rho_size));

    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ao, ao, ao_size, cudaMemcpyHostToDevice));

    int block = 128;
    int grid = (ngrid + block - 1) / block;
    get_rho_kernel<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_rho);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(rho_out, d_rho, rho_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dm));
    CUDA_CHECK(cudaFree(d_ao));
    CUDA_CHECK(cudaFree(d_rho));
}

} // extern "C"

/* ---------- end of file ---------- */
