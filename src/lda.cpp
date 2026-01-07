/*  dft.cpp  */
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <Eigen/Dense>
#include <algorithm>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::RowMajor;

extern "C" {

struct VWNPar {
    double A, b, c, x0;
};
static const VWNPar vwn_param[2] = {
    {0.0310907,  3.72744, 12.9352, -0.10498},   
    {0.01554535, 7.06042, 18.0578, -0.32500}    
};

inline void lda_exc_vxc_impl(int n, const double *rho,
                             double *exc, double *vxc, double zeta)
{
    const double pi = 3.14159265358979323846;
    const double Cx = 0.7385587663820224;

    const double z2 = zeta * zeta;

    for (int i = 0; i < n; ++i) {
        double r  = std::max(rho[i], 1e-300);
        double rs = std::pow(3.0 / (4.0 * pi * r), 1.0 / 3.0);
        double x  = std::sqrt(rs);

        double ec0, dec0_dx, ec1, dec1_dx;
        auto vwn_ec = [](double x, const VWNPar &p, double &ec, double &dec_dx)
        {
            const double X = x * x + p.b * x + p.c;
            const double Q = std::sqrt(4.0 * p.c - p.b * p.b);
            const double log_term  = std::log(x * x / X);
            const double atan_term = 2.0 * p.b / Q * std::atan(Q / (2.0 * x + p.b));
            const double x02 = p.x0 * p.x0;
            const double denom = x02 + p.b * p.x0 + p.c;
            const double corr  = p.b * p.x0 / denom *
                (std::log((x - p.x0) * (x - p.x0) / X) +
                 2.0 * (2.0 * p.x0 + p.b) / Q * std::atan(Q / (2.0 * x + p.b)));
            ec = p.A * (log_term + atan_term - corr);
            dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                            p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
        };

        vwn_ec(x, vwn_param[0], ec0, dec0_dx);
        vwn_ec(x, vwn_param[1], ec1, dec1_dx);

        double ec      = ec0      + (ec1      - ec0)      * z2;
        double dec_dx  = dec0_dx  + (dec1_dx  - dec0_dx)  * z2;
        double vc      = ec - rs / 3.0 * dec_dx / (2.0 * x);

        double rho13 = std::pow(r, 1.0 / 3.0);
        double ex    = -Cx * r * rho13;
        double vx    = -4.0 / 3.0 * Cx * rho13;

        exc[i] = ex + r * ec;
        vxc[i] = vx + vc;
    }
}

void lda_exc_vxc(int n, const double *rho, double *exc, double *vxc)
{
    lda_exc_vxc_impl(n, rho, exc, vxc, 0.0);   
}

} 





extern "C"
void build_vxc_matrix(int nao, int ngrid,
                      const double *ao,     
                      const double *weights,
                      const double *rho,
                      double *vxc_mat)      
{
    double *exc_buf = (double*)malloc(sizeof(double)*ngrid);
    double *vxc_buf = (double*)malloc(sizeof(double)*ngrid);

    lda_exc_vxc(ngrid, rho, exc_buf, vxc_buf);

    std::memset(vxc_mat, 0, sizeof(double)*nao*nao);

    for (int g = 0; g < ngrid; ++g) {
        double wv = weights[g] * vxc_buf[g];
        for (int i = 0; i < nao; ++i) {
            double aoi = ao[g*nao + i];
            for (int j = 0; j < nao; ++j) {
                vxc_mat[i*nao + j] += wv * aoi * ao[g*nao + j];
            }
        }
    }
    free(exc_buf);
    free(vxc_buf);
}


extern "C"
double compute_exc_energy(int ngrid,
                          const double *weights,
                          const double *rho)
{
    double *exc = (double*)malloc(sizeof(double)*ngrid);
    double *vxc = (double*)malloc(sizeof(double)*ngrid);

    lda_exc_vxc(ngrid, rho, exc, vxc);

    double exc_sum = 0.0;
    for (int g = 0; g < ngrid; ++g)
        exc_sum += weights[g] * exc[g];

    free(exc);
    free(vxc);
    return exc_sum;
}


extern "C"
void build_coulomb_matrix(int nao,
                          const double *eri,  
                          const double *dm,   
                          double *J)           
{
    std::memset(J, 0, sizeof(double)*nao*nao);
    for (int m = 0; m < nao; ++m)
        for (int n = 0; n < nao; ++n)
            for (int l = 0; l < nao; ++l)
                for (int s = 0; s < nao; ++s)
                    J[m*nao + n] += dm[l*nao + s] * eri[((m*nao+n)*nao+l)*nao+s];
}

extern "C" {
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
}


extern "C" {
void get_rho(int nao, int ngrid,
             const double *dm,
             const double *ao,
             double *rho_out)
{
    std::memset(rho_out, 0, sizeof(double) * ngrid);

    for (int g = 0; g < ngrid; ++g) {
        const double *phi_g = ao + g * nao;   
        for (int u = 0; u < nao; ++u) {
            for (int v = 0; v < nao; ++v) {
                rho_out[g] += dm[u * nao + v] * phi_g[u] * phi_g[v];
            }
        }
    }
}
}
