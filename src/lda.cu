#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <iostream>

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error at line %d\n",__LINE__); } }while(0)

#define RHO_EPS 1e-20

// --- VWN Constants ---
struct VWNPar {
    double A, b, c, x0;
};

__constant__ VWNPar vwn_param[2];


const VWNPar vwn_param_host[2] = {
    {0.0310907,  3.72744, 12.9352, -0.10498},   
    {0.01554535, 7.06042, 18.0578, -0.32500}   
};



__device__ inline void slater_exchange(double rho, double &ex, double &vx) {
    if (rho < RHO_EPS) { ex = 0.0; vx = 0.0; return; }
    const double Cx = 0.7385587663820224;
    double rho13 = pow(rho, 1.0/3.0);

    ex = -Cx * rho13;

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
    
    dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                    p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
}

__device__ inline void vwn_correlation(double rho, double &ec, double &vc) {
    if (rho < RHO_EPS) { ec = 0.0; vc = 0.0; return; }
    
    const double pi = 3.14159265358979323846;
    double rs = pow(3.0 / (4.0 * pi * rho), 1.0 / 3.0);
    double x  = sqrt(rs);

    double ec0, dec0_dx; 

    vwn_ec_sub(x, vwn_param[0], ec0, dec0_dx);
    
    ec = ec0;
    

    vc = ec - (rs / 3.0) * (dec0_dx / (2.0 * x));
}


__global__ void get_rho_kernel(int nao, int ngrid, const double *dm, const double *ao, double *rho_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;

    const double *phi = ao + (size_t)g * nao; 
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


__global__ void lda_fused_kernel(
    int ngrid, int nao, bool compute_B,
    const double *w, const double *rho, const double *ao,
    double *exc_out, double *B_mat
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
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

    if (!compute_B && exc_out) {
        exc_out[k] = r_val * (ex + ec); 
        return; 
    }


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
static bool vwn_init = false;


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
    
    // double h[3] = {0};
    // double h_stride = 0.0;
    // cudaMemcpy(h,       d_ao,          3*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_stride, d_ao + nao,  sizeof(double),   cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // std::cout << "[DEBUG] d_ao[0~2] = "
    //           << h[0] << ", " << h[1] << ", " << h[2] << '\n'
    //           << "[DEBUG] d_ao[nao] = " << h_stride << std::endl;
    
    if (!handle) cublasCreate(&handle);
    check_vwn_init();

    int block = 256;
    int grid = (ngrid + block - 1) / block;
    
    lda_fused_kernel<<<grid, block>>>(
        ngrid, nao, true, 
        d_weights, d_rho, d_ao, 
        nullptr, d_B_work
    );


    
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nao, nao, ngrid,
        &alpha,
        d_ao, nao,      
        d_B_work, nao, 
        &beta,
        d_vxc, nao   
    );
}

double compute_exc_gpu(int ngrid, int nao, const double *d_weights, 
                       const double *d_rho, double *d_exc_work)
{
    check_vwn_init();
    
    int block = 256;
    int grid = (ngrid + block - 1) / block;

    lda_fused_kernel<<<grid, block>>>(
        ngrid, nao, false, 
        nullptr, d_rho, nullptr, 
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