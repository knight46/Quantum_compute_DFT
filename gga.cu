/*
 * gga.cu – PBE-GGA exchange+correlation (Corrected)
 * nvcc -O3 -std=c++14 -Xcompiler -fPIC -shared gga.cu -o libgga.so
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::RowMajor;

#define CUDA_CHECK(call)                                                    \
do{                                                                         \
    cudaError_t err = (call);                                               \
    if(err != cudaSuccess){                                                 \
        fprintf(stderr,"CUDA error %s:%d  code=%d '%s'\n",                  \
                __FILE__,__LINE__,(int)err,cudaGetErrorString(err));        \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
}while(0)

struct VWNPar{ double A,b,c,x0; };
__constant__ VWNPar vwn_param[2];

__device__ inline double atomicAdd_double(double *address,double val){
#if __CUDA_ARCH__>=600
    return atomicAdd(address,val);
#else
    typedef unsigned long long int ull;
    ull* addr_as_ull=(ull*)address;
    ull old=*addr_as_ull,assumed;
    do{
        assumed=old;
        old=atomicCAS(addr_as_ull,assumed,
                      __double_as_longlong(__longlong_as_double(assumed)+val));
    }while(assumed!=old);
    return __longlong_as_double(old);
#endif
}

/* ---------- Device Functions: PBE/VWN ---------- */

__device__ inline
void vwn_ec_device(double x,const VWNPar &p,double &ec,double &dec_dx){
    const double X = x*x + p.b*x + p.c;
    const double Q = sqrt(4.0*p.c - p.b*p.b);
    const double log_term  = log(x*x/X);
    const double atan_term = 2.0*p.b/Q * atan(Q/(2.0*x+p.b));
    const double x02=p.x0*p.x0;
    const double denom=x02+p.b*p.x0+p.c;
    const double corr=p.b*p.x0/denom*
        (log((x-p.x0)*(x-p.x0)/X)+2.0*(2.0*p.x0+p.b)/Q*atan(Q/(2.0*x+p.b)));
    ec=p.A*(log_term+atan_term-corr);
    dec_dx=p.A*(2.0/x-(2.0*x+p.b)/X-
               p.b*p.x0/denom*(2.0/(x-p.x0)-(2.0*x+p.b)/X));
}

__device__ inline
void pbe_exchange(double rho,double sigma,
                  double &ex,double &vrho,double &vsigma){
    // Numerical safety
    if(rho < 1e-15) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
    
    const double pi=M_PI;
    const double Cx=-0.7385587663820224; 
    const double kappa=0.804;
    const double mu=0.2195149727645171; 

    double rho13=pow(rho,1.0/3.0);
    double kF=pow(3.0*pi*pi*rho,1.0/3.0);
    
    // Avoid division by zero if gradient is zero
    double s = (sigma > 1e-24) ? sqrt(sigma)/(2.0*kF*rho) : 0.0;
    
    double denom=1.0+mu*s*s/kappa;
    double F=1.0+kappa-kappa/denom;

    ex=Cx*rho13*F;
    double dFds=2.0*mu*s/(denom*denom);
    vrho=Cx*(4.0/3.0*rho13*F - rho13*dFds*s);
    vsigma=0.5*Cx*rho13*dFds/(2.0*kF*rho);
}

__device__ inline
void pbe_correlation(double rho,double sigma,
                     double &ec,double &vrho,double &vsigma){
    if(rho < 1e-15) { ec=0.0; vrho=0.0; vsigma=0.0; return; }

    // LDA part
    double x=sqrt(pow(3.0/(4.0*M_PI*rho),1.0/3.0));
    double ec0,dec0_dx,ec1,dec1_dx;
    vwn_ec_device(x,vwn_param[0],ec0,dec0_dx);
    vwn_ec_device(x,vwn_param[1],ec1,dec1_dx);
    double zeta=0.0,z2=zeta*zeta;
    double ec_lda=ec0+(ec1-ec0)*z2;
    double dec_lda_dx = dec0_dx + (dec1_dx - dec0_dx)*z2;
    double vc_lda=ec_lda - x/6.0 * dec_lda_dx;

    // Gradient correction
    const double beta=0.066725;
    const double gamma=(1.0-log(2.0))/(M_PI*M_PI);
    double kF=pow(3.0*M_PI*M_PI*rho,1.0/3.0);
    double t = (sigma > 1e-24) ? sqrt(sigma)/(2.0*kF*rho)*pow(3.0*M_PI*M_PI,-1.0/3.0) : 0.0;
    
    double A=beta/(gamma*(exp(-ec_lda/gamma)-1.0));
    double denom=1.0+A*t*t;
    double H=gamma*log(1.0+beta/gamma*t*t*denom);

    ec=ec_lda+H;
    double dHdt=2.0*beta*t*(1.0+2.0*A*t*t)/(denom*denom);
    vrho=vc_lda - (H - t*dHdt)/3.0;
    vsigma=0.5*dHdt/(2.0*kF*rho*pow(3.0*M_PI*M_PI,1.0/3.0));
}

/* ---------- Kernels ---------- */

__global__ void gga_exc_vxc_kernel(int ngrid,
                                   const double *rho,
                                   const double *sigma,
                                   double *exc,
                                   double *vrho,
                                   double *vsigma){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=ngrid) return;
    
    // Rename to avoid shadowing
    double r_val=rho[g];
    double s_val=sigma[g];
    
    if(r_val < 1e-15) r_val = 1e-15;
    if(s_val < 0.0) s_val = 0.0;

    // Debug print only for first thread
    // if(g==0){
    //     double kF = pow(3.*M_PI*M_PI*r_val, 1./3.);
    //     double s_param = sqrt(s_val) / (2.*kF*r_val);
    //     printf("g0: rho=%.4e  sigma=%.4e  s=%.4e  kF=%.4e\n",
    //          r_val, s_val, s_param, kF);
    // }
                      
    double ex, vrx, vsx;
    double ec, vrc, vsc;
    pbe_exchange(r_val, s_val, ex, vrx, vsx);
    pbe_correlation(r_val, s_val, ec, vrc, vsc);

    // FIXED: output energy density (rho * epsilon) for correct integration
    if(exc)    exc[g]    = r_val * (ex + ec); 
    if(vrho)   vrho[g]   = vrx + vrc;
    if(vsigma) vsigma[g] = vsx + vsc;
}

__global__ void get_rho_sigma_kernel(int nao, int ngrid,
                                     const double *dm,
                                     const double *ao,
                                     const double *ao_grad,
                                     double *rho,
                                     double *sigma,
                                     double *grad_rho_out) // Added output
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;

    const double *phi = ao + (size_t)g * nao;
    double r = 0.0;
    for (int u = 0; u < nao; ++u) {
        double phiu = phi[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v) r += dm_row[v] * phiu * phi[v];
    }
    rho[g] = r;

    double grd[3] = {0.0, 0.0, 0.0};
    const double *gphi = ao_grad + (size_t)g * (3 * nao);
    for (int d = 0; d < 3; ++d) {
        for (int u = 0; u < nao; ++u) {
            double phi_u    = phi[u];
            double gphi_du = gphi[d * nao + u];
            const double *dm_row = dm + (size_t)u * nao;
            for (int v = 0; v < nao; ++v)
                grd[d] += dm_row[v] * (gphi_du * phi[v] + phi_u * gphi[d * nao + v]);
        }
    }
    sigma[g] = grd[0]*grd[0] + grd[1]*grd[1] + grd[2]*grd[2];
    
    // Store Gradient Vector
    if(grad_rho_out){
        grad_rho_out[g*3+0] = grd[0];
        grad_rho_out[g*3+1] = grd[1];
        grad_rho_out[g*3+2] = grd[2];
    }
}

__global__ void build_vxc_matrix_gga_kernel(int nao,int rows,int g0,
                                            const double *ao_b,
                                            const double *ao_grad_b,
                                            const double *w_b,
                                            const double *vrho_b,
                                            const double *vsigma_b,
                                            const double *grad_rho_b, // Added input
                                            double *vxc_mat){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=rows*nao) return;
    int im=idx/nao;
    int i =idx%nao;

    double aoi=ao_b[im*nao+i];
    double w=w_b[im];
    double vr=vrho_b[im], vs=vsigma_b[im];
    
    // Retrieve grad_rho for this grid point
    double gr[3];
    gr[0] = grad_rho_b[im*3+0];
    gr[1] = grad_rho_b[im*3+1];
    gr[2] = grad_rho_b[im*3+2];

    // Calculate Dot(Grad_rho, Grad_phi_i)
    double dot_rho_phi_i = 0.0;
    const double *gbase_i = ao_grad_b + im*(3*nao);
    for(int d=0; d<3; ++d) dot_rho_phi_i += gr[d] * gbase_i[d*nao+i];

    for(int j=0;j<nao;++j){
        double aoj=ao_b[im*nao+j];
        
        // Calculate Dot(Grad_rho, Grad_phi_j)
        double dot_rho_phi_j = 0.0;
        for(int d=0; d<3; ++d) dot_rho_phi_j += gr[d] * gbase_i[d*nao+j];

        // GGA Vxc term: w * ( v_rho * phi_i * phi_j + 2 * v_sigma * [ phi_i * (grad_rho . grad_phi_j) + phi_j * (grad_rho . grad_phi_i) ] )
        // Using product rule expansion of: 2 * v_sigma * (grad_rho . grad(phi_i * phi_j))
        double term_rho = vr * aoi * aoj;
        double term_sig = 2.0 * vs * (aoi * dot_rho_phi_j + aoj * dot_rho_phi_i);
        
        atomicAdd_double(&vxc_mat[i*nao+j], w * (term_rho + term_sig));
    }
}

__global__ void build_coulomb_kernel(int nao, int rows_m, int m0,
                                     const double *eri_slice,
                                     const double *dm,
                                     double *J)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = rows_m * nao;
    if (idx >= tot) return;
    int im = idx / nao;
    int n  = idx % nao;
    int m  = m0 + im;

    double sum = 0.0;
    for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
            size_t pos = ((size_t)im * nao + n) * nao * nao + (size_t)l * nao + s;
            sum += dm[l * nao + s] * eri_slice[pos];
        }
    }
    atomicAdd_double(&J[m * nao + n], sum);
}

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

__global__ void weighted_sum_gga_kernel(const double *w,const double *exc,double *out,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    // exc[tid] already contains rho * epsilon
    atomicAdd_double(out,w[tid]*exc[tid]);
}

extern "C" {

static void copy_vwn_once(){
    static bool done=false;
    if(!done){
        VWNPar h[2]={
            {0.0310907, 3.72744,12.9352,-0.10498},
            {0.01554535,7.06042,18.0578,-0.32500}
        };
        CUDA_CHECK(cudaMemcpyToSymbol(vwn_param,h,sizeof(VWNPar)*2));
        done=true;
    }
}

void build_vxc_matrix_gga(int nao,int ngrid,
                          const double *ao,
                          const double *ao_grad,
                          const double *weights,
                          const double *rho,
                          const double *sigma,
                          const double *grad_rho, // Added Argument
                          double *vxc_mat)
{
    copy_vwn_once();
    size_t free_byte,total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte,&total_byte));
    const size_t SAFE=size_t(free_byte*0.9);
    const size_t aux=64<<20;
    // Memory calculation: ao + grad + w + rho + sig + grad_rho + vr + vs
    const size_t per_row=(nao*4+5+3)*sizeof(double); 
    size_t rows_per=(SAFE>aux)?(SAFE-aux)/per_row:0;
    if(rows_per==0){ fprintf(stderr,"GGA: not enough GPU memory\n"); exit(1); }
    if(rows_per>ngrid) rows_per=ngrid;

    double *d_ao=0,*d_grad=0,*d_w=0,*d_rho=0,*d_sig=0,*d_grho=0,*d_vr=0,*d_vs=0,*d_mat=0;
    CUDA_CHECK(cudaMalloc(&d_ao,   rows_per*nao*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, rows_per*3*nao*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,    rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho,  rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig,  rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grho, rows_per*3*sizeof(double))); // Alloc grad_rho
    CUDA_CHECK(cudaMalloc(&d_vr,   rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vs,   rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mat,  (size_t)nao*nao*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mat,0,(size_t)nao*nao*sizeof(double)));

    const int BLOCK=256;
    for(int g0=0;g0<ngrid;g0+=rows_per){
        int g1=std::min(g0+(int)rows_per,ngrid);
        int rows=g1-g0;

        CUDA_CHECK(cudaMemcpyAsync(d_ao,   ao   +(size_t)g0*nao,
                                   rows*nao*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_grad, ao_grad+(size_t)g0*3*nao,
                                   rows*3*nao*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_w,    weights+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho,  rho+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig,  sigma+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_grho, grad_rho+g0*3,
                                   rows*3*sizeof(double),cudaMemcpyHostToDevice));

        int grid=(rows+BLOCK-1)/BLOCK;
        gga_exc_vxc_kernel<<<grid,BLOCK>>>(rows,d_rho,d_sig,nullptr,d_vr,d_vs);
        CUDA_CHECK(cudaGetLastError());

        int N=rows*nao;
        int grid2=(N+BLOCK-1)/BLOCK;
        build_vxc_matrix_gga_kernel<<<grid2,BLOCK>>>(
                nao,rows,g0,d_ao,d_grad,d_w,d_vr,d_vs,d_grho,d_mat);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(vxc_mat,d_mat,(size_t)nao*nao*sizeof(double),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_grho));
    CUDA_CHECK(cudaFree(d_vr)); CUDA_CHECK(cudaFree(d_vs)); CUDA_CHECK(cudaFree(d_mat));
}

double compute_exc_energy_gga(int ngrid,
                              const double *weights,
                              const double *rho,
                              const double *sigma)
{
    copy_vwn_once();
    size_t free_byte,total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte,&total_byte));
    const size_t SAFE=size_t(free_byte*0.9);
    const size_t aux=64<<20;
    const size_t per_row=4*sizeof(double);
    size_t rows_per=(SAFE>aux)?(SAFE-aux)/per_row:0;
    if(rows_per==0){ fprintf(stderr,"GGA energy: not enough GPU mem\n"); exit(1); }
    if(rows_per>ngrid) rows_per=ngrid;

    double *d_w=0,*d_rho=0,*d_sig=0,*d_exc=0,*d_sum=0;
    CUDA_CHECK(cudaMalloc(&d_w,   rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, rows_per*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));

    const int BLOCK=256;
    double exc_total=0.0;
    for(int g0=0;g0<ngrid;g0+=rows_per){
        int g1=std::min(g0+(int)rows_per,ngrid);
        int rows=g1-g0;

        CUDA_CHECK(cudaMemcpyAsync(d_w,   weights+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_rho, rho+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_sig, sigma+g0,
                                   rows*sizeof(double),cudaMemcpyHostToDevice));

        int grid=(rows+BLOCK-1)/BLOCK;
        gga_exc_vxc_kernel<<<grid,BLOCK>>>(rows,d_rho,d_sig,d_exc,nullptr,nullptr);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_sum,0,sizeof(double)));
        weighted_sum_gga_kernel<<<grid,BLOCK>>>(d_w,d_exc,d_sum,rows);
        CUDA_CHECK(cudaGetLastError());

        double partial=0.0;
        CUDA_CHECK(cudaMemcpy(&partial,d_sum,sizeof(double),cudaMemcpyDeviceToHost));
        exc_total+=partial;
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_exc)); CUDA_CHECK(cudaFree(d_sum));
    return exc_total;
}

void get_rho_sigma(int nao, int ngrid,
                   const double *dm,
                   const double *ao,
                   const double *ao_grad,
                   double *rho,
                   double *sigma,
                   double *grad_rho) // Added Argument
{
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = size_t(free_byte * 0.9);

    const size_t dm_bytes  = (size_t)nao * nao * sizeof(double);
    const size_t row_bytes = (size_t)nao * sizeof(double);
    const size_t rho_bytes = sizeof(double);
    const size_t gr_bytes  = 3 * sizeof(double);
    const size_t aux_buf   = 64 << 20;
    const size_t left_bytes = (SAFE_FREE > dm_bytes + aux_buf) ?
                              (SAFE_FREE - dm_bytes - aux_buf) : 0;
    if (left_bytes == 0) {
        std::cerr << "Not enough GPU memory to tile get_rho_sigma!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t rows_per = left_bytes / (row_bytes * 4 + rho_bytes + gr_bytes);
    if (rows_per < 1) rows_per = 1;
    if (rows_per > ngrid) rows_per = ngrid;

    double *d_dm = nullptr;
    double *d_ao = nullptr, *d_grad = nullptr;
    double *d_rho = nullptr, *d_sig = nullptr, *d_grho = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dm,  dm_bytes));
    CUDA_CHECK(cudaMalloc(&d_ao,  rows_per * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad,rows_per * 3 * nao * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho, rows_per * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sig, rows_per * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grho,rows_per * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 128;
    for (int g0 = 0; g0 < ngrid; g0 += rows_per) {
        int g1 = std::min(g0 + (int)rows_per, ngrid);
        int rows = g1 - g0;

        CUDA_CHECK(cudaMemcpyAsync(d_ao,   ao   + (size_t)g0 * nao,
                                   rows * nao * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_grad, ao_grad + (size_t)g0 * 3 * nao,
                                   rows * 3 * nao * sizeof(double), cudaMemcpyHostToDevice));

        int grid = (rows + BLOCK - 1) / BLOCK;
        get_rho_sigma_kernel<<<grid, BLOCK>>>(nao, rows, d_dm, d_ao, d_grad,
                                              d_rho, d_sig, d_grho);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(rho   + g0, d_rho, rows * sizeof(double),
                                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(sigma + g0, d_sig, rows * sizeof(double),
                                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(grad_rho + g0*3, d_grho, rows * 3 * sizeof(double),
                                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_ao)); CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_rho)); CUDA_CHECK(cudaFree(d_sig)); CUDA_CHECK(cudaFree(d_grho));
}

void build_coulomb_matrix(int nao,
                          const double *eri,
                          const double *dm,
                          double *J)
{
    size_t free_byte = 0, total_byte = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    const size_t SAFE_FREE = free_byte * 0.9;

    const size_t dm_bytes   = (size_t)nao * nao * sizeof(double);
    const size_t j_bytes    = (size_t)nao * nao * sizeof(double);
    const size_t aux_buf    = 128 * 1024 * 1024;
    const size_t eri_row3   = (size_t)nao * nao * nao * sizeof(double);
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
    double *d_eri_slice = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes));
    CUDA_CHECK(cudaMalloc(&d_J,  j_bytes));
    CUDA_CHECK(cudaMalloc(&d_eri_slice, block_m * eri_row3));

    CUDA_CHECK(cudaMemcpy(d_dm, dm, dm_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J, 0, j_bytes));

    const int BLOCK = 256;
    for (int m0 = 0; m0 < nao; m0 += block_m) {
        int m1 = std::min(m0 + (int)block_m, nao);
        int rows = m1 - m0;

        CUDA_CHECK(cudaMemcpyAsync(d_eri_slice,
                                   eri + (size_t)m0 * nao * nao * nao,
                                   (size_t)rows * eri_row3,
                                   cudaMemcpyHostToDevice));

        int grid = ((rows * nao) + BLOCK - 1) / BLOCK;
        build_coulomb_kernel<<<grid, BLOCK>>>(
                nao, rows, m0, d_eri_slice, d_dm, d_J);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(J, d_J, j_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_J)); CUDA_CHECK(cudaFree(d_eri_slice));
}

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

    CUDA_CHECK(cudaFree(d_dm)); CUDA_CHECK(cudaFree(d_ao_block)); CUDA_CHECK(cudaFree(d_rho_block));
}

} // extern "C"