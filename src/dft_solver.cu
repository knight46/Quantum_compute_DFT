#include "dft_solver.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#define CUDA_CHECK(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error: %s:%d\n",__FILE__,__LINE__); } }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t err=(call); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"CUBLAS Error at line %d\n",__LINE__); } }while(0)

#define RHO_EPS 1e-12
#define MIN_GRAD 1e-20

namespace kernels {

    struct VWNPar { double A, b, c, x0; };
    __constant__ VWNPar c_vwn_param[2];
    __constant__ double c_A_pw92;
    
    const VWNPar h_vwn_param[2] = {
        {0.0310907,  3.72744, 12.9352, -0.10498},  
        {0.01554535, 7.06042, 18.0578, -0.32500}   
    };
    const double h_A_pw92 = 0.03109069086965489503;

    __constant__ double alpha1 = 0.21370;
    __constant__ double beta1  = 7.5957;
    __constant__ double beta2  = 3.5876;
    __constant__ double beta3  = 1.6382;
    __constant__ double beta4  = 0.49294;

    __constant__ double C_LDA_X = 0.80;      
    __constant__ double C_B88_X = 0.72;
    __constant__ double C_VWN_C = 0.19;      
    __constant__ double C_LYP_C = 0.81;

    __constant__ double A_VWN_B3 = 0.0310907;
    __constant__ double b_VWN_B3 = 13.0720;
    __constant__ double c_VWN_B3 = 42.7198;
    __constant__ double x0_VWN_B3 = -0.409286;

    __constant__ double BETA_B88 = 0.0042;

    #define LYP_A 0.04918
    #define LYP_B 0.132
    #define LYP_C 0.2533
    #define LYP_D 0.349
    #define LYP_CF     2.87123400018819108 

    static bool constant_init = false;

    void init_gpu_constants() {
        if (!constant_init) {
            cudaMemcpyToSymbol(c_vwn_param, h_vwn_param, sizeof(VWNPar)*2);
            cudaMemcpyToSymbol(c_A_pw92, &h_A_pw92, sizeof(double));
            constant_init = true;
        }
    }

    __device__ inline void slater_exchange(double rho, double &ex, double &vx) {
        if (rho < RHO_EPS) { ex = 0.0; vx = 0.0; return; }
        const double Cx = 0.7385587663820224;
        double rho13 = pow(rho, 1.0/3.0);
        ex = -Cx * rho13;
        vx = (4.0/3.0) * ex;
    }

    __device__ inline void slater_exchange_b3lyp(double rho, double &ex, double &vx) {
        const double Cx = -0.7385587663820224; 
        if (rho < RHO_EPS) { ex = 0.0; vx = 0.0; return; }
        
        double rho13 = pow(rho, 1.0/3.0);
        ex = Cx * rho13;          
        vx = (4.0/3.0) * ex;      
    }

    __device__ inline void b88_exchange(double rho, double sigma, double &ex, double &vrho, double &vsigma) {
        if (rho < RHO_EPS) { ex=0.0; vrho=0.0; vsigma=0.0; return; }
        if (sigma < MIN_GRAD) { ex=0.0; vrho=0.0; vsigma=0.0; return; }

        double rho13 = pow(rho, 1.0/3.0);
        double rho43 = rho * rho13;
        double grad_rho = sqrt(sigma);

        double x = grad_rho / rho43;
        double x2 = x * x;
        double asinh_x = asinh(x);
        
        double denom = 1.0 + 6.0 * BETA_B88 * x * asinh_x;
        double term = BETA_B88 * x2 / denom;
        
        ex = -term * rho13; 

        double d_denom_dx = 6.0 * BETA_B88 * (asinh_x + x / sqrt(1.0 + x2));
        double dF_dx = BETA_B88 * (2.0 * x * denom - x2 * d_denom_dx) / (denom * denom);
        
        double dE_dx = rho43 * (-dF_dx); 
        
        vsigma = dE_dx * (1.0 / (2.0 * rho43 * grad_rho)); 
        
        double E_dens = rho43 * (-term);
        vrho = (4.0/3.0) * (E_dens / rho) - (4.0/3.0) * dE_dx * (x / rho);
    }

    __device__ inline void vwn_correlation_b3lyp(double rho, double &ec, double &vc) {
        if (rho < RHO_EPS) { ec=0.0; vc=0.0; return; }
        
        double rs = pow(3.0 / (4.0 * M_PI * rho), 1.0/3.0);
        double x = sqrt(rs);
        
        double X = x * x + b_VWN_B3 * x + c_VWN_B3;
        double Q = sqrt(4.0 * c_VWN_B3 - b_VWN_B3 * b_VWN_B3);
        
        double log_term = log(x * x / X);
        double atan_term = (2.0 / Q) * atan(Q / (2.0 * x + b_VWN_B3));
        
        double x02 = x0_VWN_B3 * x0_VWN_B3;
        double X_x0 = x02 + b_VWN_B3 * x0_VWN_B3 + c_VWN_B3; 
        
        double term_c_log = log(pow(x - x0_VWN_B3, 2.0) / X);
        double term_c_atan = (2.0 * (2.0 * x0_VWN_B3 + b_VWN_B3) / Q) * atan(Q / (2.0 * x + b_VWN_B3));
        
        double eps_c = A_VWN_B3 * ( log_term + b_VWN_B3 * atan_term - 
                                     (b_VWN_B3 * x0_VWN_B3 / X_x0) * (term_c_log + term_c_atan) );
        
        ec = eps_c;

        double d_log = 2.0/x - (2.0*x + b_VWN_B3)/X;
        double d_atan = -1.0 / X; 
        double d_term_c_log = 2.0/(x - x0_VWN_B3) - (2.0*x + b_VWN_B3)/X;
        double d_term_c_atan = -(2.0*x0_VWN_B3 + b_VWN_B3) / X;

        double deps_dx = A_VWN_B3 * ( d_log + b_VWN_B3 * d_atan - 
                                       (b_VWN_B3 * x0_VWN_B3 / X_x0) * (d_term_c_log + d_term_c_atan) );

        vc = eps_c - (rs / 3.0) * (deps_dx / (2.0 * x));
    }

    __device__ inline void lyp_correlation(
        double rho, double sigma,
        double &ec, double &vrho, double &vsigma
    ) {
        if (rho < 1e-14) {
            ec = 0.0; vrho = 0.0; vsigma = 0.0;
            return;
        }
        double r_13 = pow(rho, 1.0/3.0);      
        double r_m13 = 1.0 / r_13;            
        double r_m53 = r_m13 * r_m13 * r_m13 * r_m13 * r_m13; 
        double exp_val = exp(-LYP_C * r_m13);
        double denom = 1.0 + LYP_D * r_m13;
        double denom_inv = 1.0 / denom;
        double G = exp_val * denom_inv;
        double term_d = LYP_D * r_m13 * denom_inv;
        double delta = LYP_C * r_m13 + term_d;
        double H1 = -LYP_A * rho * denom_inv;
        double H2a = -LYP_A * LYP_B * LYP_CF * rho * G;
        double coeff_grad = (LYP_A * LYP_B / 72.0) * sigma * r_m53 * G;
        double H2b = coeff_grad * (3.0 + 7.0 * delta);
        double H = H1 + H2a + H2b;
        ec = H / rho;
        double d_rm13 = -(1.0/3.0) * r_m13 / rho;
        double d_denom = LYP_D * d_rm13;
        double d_G = G * delta / (3.0 * rho);
        double d_term_d = LYP_D * (d_rm13 * denom_inv - r_m13 * denom_inv * denom_inv * d_denom);
        double d_delta = LYP_C * d_rm13 + d_term_d;
        double d_H1 = -LYP_A * (denom - rho * d_denom) * (denom_inv * denom_inv);
        double d_H2a = -LYP_A * LYP_B * LYP_CF * (G + rho * d_G);
        double pre_factor = r_m53 * G;
        double group_bracket = 3.0 + 7.0 * delta;
        double term_deriv = (-5.0/(3.0*rho)) * group_bracket 
                          + (delta / (3.0*rho)) * group_bracket 
                          + 7.0 * d_delta;                    
        double d_H2b = (LYP_A * LYP_B / 72.0) * sigma * pre_factor * term_deriv;
        vrho = d_H1 + d_H2a + d_H2b;
        vsigma = (LYP_A * LYP_B / 72.0) * r_m53 * G * (3.0 + 7.0 * delta);
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
        vwn_ec_sub(x, c_vwn_param[0], ec0, dec0_dx);
        ec = ec0;
        vc = ec - (rs / 3.0) * (dec0_dx / (2.0 * x));
    }

    __device__ inline void pw92_correlation_rks(double rho, double &ec, double &vc) {
        if (rho < RHO_EPS) { ec = 0.0; vc = 0.0; return; }
        const double rs = pow(3.0 / (4.0 * M_PI * rho), 1.0/3.0);
        const double rs_sqrt = sqrt(rs);
        double Q = 2.0 * c_A_pw92 * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs * rs_sqrt + beta4 * rs * rs);
        double Q_prime = 2.0 * c_A_pw92 * (0.5 * beta1 / rs_sqrt + beta2 + 1.5 * beta3 * rs_sqrt + 2.0 * beta4 * rs);
        double log_term = log(1.0 + 1.0 / Q);
        double f_rs = -2.0 * c_A_pw92 * (1.0 + alpha1 * rs);
        ec = f_rs * log_term;
        double df_drs = -2.0 * c_A_pw92 * alpha1;
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
        double dx_drho = (vc_lda - ec_lda) / (rho * gamma); 
        double exp_x = exp(x);
        double dA_dx = -A * exp_x / expm1_x;
        double dA_drho = dA_dx * dx_drho;
        double dt2_drho = t2 * (-7.0/3.0) / rho;
        vrho = vc_lda + H + rho * (dH_dA * dA_drho + dH_dt2 * dt2_drho);
    }

    __global__ void reduce_sum_kernel(const double *w, const double *val, double *out, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        double sum = 0.0;
        for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
            sum += w[i] * val[i];
        }
        atomicAdd(out, sum);
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
    ) {
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

        double ex=0, vx=0, ec=0, vc=0;
        slater_exchange(r_val, ex, vx);
        vwn_correlation(r_val, ec, vc);

        if (!compute_B && exc_out) {
            exc_out[k] = r_val * (ex + ec); 
            return; 
        }

        if (compute_B) {
            double v_total = vx + vc;
            double factor = w[k] * v_total;
            const double *phi_row = ao + k * nao;
            double *B_row = B_mat + k * nao;
            for (int i = 0; i < nao; ++i) {
                B_row[i] = factor * phi_row[i];
            }
        }
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
    ) {
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
        slater_exchange_b3lyp(r_val, ex_lda, vx_lda);
        
        double r_val_spin = r_val * 0.5;
        double s_val_spin = s_val * 0.25;

        double ex_b88=0, vrx_b88=0, vsx_b88=0;
        double ex_b88_tmp, vrx_b88_tmp, vsx_b88_tmp;
        
        b88_exchange(r_val_spin, s_val_spin, ex_b88_tmp, vrx_b88_tmp, vsx_b88_tmp);
        
        ex_b88 = ex_b88_tmp; 
        vrx_b88 = vrx_b88_tmp; 
        vsx_b88 = 0.5 * vsx_b88_tmp; 

        double ec_vwn=0, vc_vwn=0;
        vwn_correlation_b3lyp(r_val, ec_vwn, vc_vwn);

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
            
            total_vrho *= 0.5;

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

    __global__ void symmetrize_matrix_kernel(int n, double *mat) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (r < n && c <= r) {
            int idx1 = r * n + c;
            int idx2 = c * n + r;
            
            double val = mat[idx1] + mat[idx2];
            mat[idx1] = val;
            mat[idx2] = val;
        }
    }
} 

struct CublasHandleWrapper {
    cublasHandle_t handle;
    CublasHandleWrapper() { cublasCreate(&handle); }
    ~CublasHandleWrapper() { cublasDestroy(handle); }
};

XCSolver::XCSolver() : handle_wrapper(new CublasHandleWrapper()) {
    kernels::init_gpu_constants();
}
XCSolver::~XCSolver() = default;

void XCSolver::safe_cublas_dgemm(bool transA, bool transB, int m, int n, int k, 
                                 const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle_wrapper->handle,
        transA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transB ? CUBLAS_OP_T : CUBLAS_OP_N,
        m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void XCSolver::compute_coulomb(int nao, const double* d_eri, const double* d_dm, double* d_J) {
    int N2 = nao * nao;
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(handle_wrapper->handle, CUBLAS_OP_N, 
                N2, N2, &alpha, d_eri, N2, d_dm, 1, &beta, d_J, 1);
}

LDASolver::LDASolver() : XCSolver() {}

double LDASolver::compute_xc(int ngrid, int nao, const double* d_dm, const double* d_ao, 
                             const double* d_ao_grad, const double* d_weights, double* d_vxc) {
    double *d_rho; CUDA_CHECK(cudaMalloc(&d_rho, ngrid * sizeof(double)));
    
    int block = 256;
    int grid = (ngrid + block - 1) / block;

    kernels::get_rho_kernel<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_rho);

    double *d_exc_work; CUDA_CHECK(cudaMalloc(&d_exc_work, ngrid * sizeof(double)));
    kernels::lda_fused_kernel<<<grid, block>>>(ngrid, nao, false, nullptr, d_rho, nullptr, d_exc_work, nullptr);
    
    double *d_sum; CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    kernels::reduce_sum_kernel<<<256, 256>>>(d_weights, d_exc_work, d_sum, ngrid);
    double exc_val;
    CUDA_CHECK(cudaMemcpy(&exc_val, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    double *d_B; CUDA_CHECK(cudaMalloc(&d_B, ngrid * nao * sizeof(double)));
    kernels::lda_fused_kernel<<<grid, block>>>(ngrid, nao, true, d_weights, d_rho, d_ao, nullptr, d_B);

    safe_cublas_dgemm(false, true, nao, nao, ngrid, d_ao, nao, d_B, nao, d_vxc, nao);

    cudaFree(d_rho); cudaFree(d_exc_work); cudaFree(d_sum); cudaFree(d_B);
    return exc_val;
}

GGASolver::GGASolver() : XCSolver() {}

double GGASolver::compute_xc(int ngrid, int nao, const double* d_dm, const double* d_ao, 
                             const double* d_ao_grad, const double* d_weights, double* d_vxc) {
    double *d_rho, *d_sigma, *d_grad_rho;
    CUDA_CHECK(cudaMalloc(&d_rho, ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma, ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_rho, 3 * ngrid * sizeof(double)));

    const double *d_gx = d_ao_grad;
    const double *d_gy = d_ao_grad + ngrid * nao;
    const double *d_gz = d_ao_grad + 2 * ngrid * nao;

    int block = 256;
    int grid = (ngrid + block - 1) / block;

    kernels::get_rho_sigma_kernel_planar<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_gx, d_gy, d_gz, d_rho, d_sigma, d_grad_rho);

    double *d_exc_work; CUDA_CHECK(cudaMalloc(&d_exc_work, ngrid * sizeof(double)));
    kernels::gga_fused_kernel<<<grid, block>>>(ngrid, nao, false, nullptr, d_rho, d_sigma, nullptr, nullptr, nullptr, nullptr, nullptr, d_exc_work, nullptr);
    
    double *d_sum; CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    kernels::reduce_sum_kernel<<<256, 256>>>(d_weights, d_exc_work, d_sum, ngrid);
    double exc_val;
    CUDA_CHECK(cudaMemcpy(&exc_val, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    double *d_B; CUDA_CHECK(cudaMalloc(&d_B, ngrid * nao * sizeof(double)));
    kernels::gga_fused_kernel<<<grid, block>>>(ngrid, nao, true, d_weights, d_rho, d_sigma, d_grad_rho, d_ao, d_gx, d_gy, d_gz, nullptr, d_B);

    safe_cublas_dgemm(false, true, nao, nao, ngrid, d_ao, nao, d_B, nao, d_vxc, nao);

    cudaFree(d_rho); cudaFree(d_sigma); cudaFree(d_grad_rho); 
    cudaFree(d_exc_work); cudaFree(d_sum); cudaFree(d_B);
    return exc_val;
}

B3LYPSolver::B3LYPSolver() : XCSolver() {}

double B3LYPSolver::compute_xc(int ngrid, int nao, const double* d_dm, const double* d_ao, 
                      const double* d_ao_grad, const double* d_weights, double* d_vxc) {
    double *d_rho, *d_sigma, *d_grad_rho;
    CUDA_CHECK(cudaMalloc(&d_rho, ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma, ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_rho, 3 * ngrid * sizeof(double)));

    const double *d_gx = d_ao_grad;
    const double *d_gy = d_ao_grad + ngrid * nao;
    const double *d_gz = d_ao_grad + 2 * ngrid * nao;

    int block = 256;
    int grid = (ngrid + block - 1) / block;

    kernels::get_rho_sigma_kernel_planar<<<grid, block>>>(nao, ngrid, d_dm, d_ao, d_gx, d_gy, d_gz, d_rho, d_sigma, d_grad_rho);

    double *d_exc_work; CUDA_CHECK(cudaMalloc(&d_exc_work, ngrid * sizeof(double)));
    kernels::b3lyp_fused_kernel<<<grid, block>>>(
        ngrid, nao, false, 
        nullptr, d_rho, d_sigma, nullptr, 
        nullptr, nullptr, nullptr, nullptr, 
        d_exc_work, nullptr
    );
    
    double *d_sum; CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    kernels::reduce_sum_kernel<<<256, 256>>>(d_weights, d_exc_work, d_sum, ngrid);
    double exc_val;
    CUDA_CHECK(cudaMemcpy(&exc_val, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    double *d_B; CUDA_CHECK(cudaMalloc(&d_B, ngrid * nao * sizeof(double)));
    kernels::b3lyp_fused_kernel<<<grid, block>>>(
        ngrid, nao, true, 
        d_weights, d_rho, d_sigma, d_grad_rho, 
        d_ao, d_gx, d_gy, d_gz, 
        nullptr, d_B
    );

    safe_cublas_dgemm(false, true, nao, nao, ngrid, d_ao, nao, d_B, nao, d_vxc, nao);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((nao+15)/16, (nao+15)/16);
    kernels::symmetrize_matrix_kernel<<<dimGrid, dimBlock>>>(nao, d_vxc);

    cudaFree(d_rho); cudaFree(d_sigma); cudaFree(d_grad_rho); 
    cudaFree(d_exc_work); cudaFree(d_sum); cudaFree(d_B);
    return exc_val;
}


extern "C" {
    
    XCSolver* DFT_CreateSolver(int type) {
        if (type == SOLVER_LDA) return new LDASolver();
        if (type == SOLVER_GGA) return new GGASolver();
        if (type == SOLVER_B3LYP) return new B3LYPSolver();
        return nullptr;
    }

    void DFT_DestroySolver(XCSolver* solver) {
        if (solver) delete solver;
    }

    double DFT_ComputeXC(XCSolver* solver, int ngrid, int nao,
                          unsigned long long d_dm_ptr,
                          unsigned long long d_ao_ptr,
                          unsigned long long d_ao_grad_ptr,
                          unsigned long long d_weights_ptr,
                          unsigned long long d_vxc_ptr) 
    {
        if (!solver) return 0.0;
        return solver->compute_xc(
            ngrid, nao,
            reinterpret_cast<const double*>(d_dm_ptr),
            reinterpret_cast<const double*>(d_ao_ptr),
            reinterpret_cast<const double*>(d_ao_grad_ptr),
            reinterpret_cast<const double*>(d_weights_ptr),
            reinterpret_cast<double*>(d_vxc_ptr)
        );
    }

    void DFT_ComputeCoulomb(XCSolver* solver, int nao,
                            unsigned long long d_eri_ptr,
                            unsigned long long d_dm_ptr,
                            unsigned long long d_J_ptr)
    {
        if (!solver) return;
        solver->compute_coulomb(
            nao,
            reinterpret_cast<const double*>(d_eri_ptr),
            reinterpret_cast<const double*>(d_dm_ptr),
            reinterpret_cast<double*>(d_J_ptr)
        );
    }
}