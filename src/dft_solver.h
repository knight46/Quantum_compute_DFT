#pragma once
#include <vector>
#include <memory>

struct CublasHandleWrapper; 

class XCSolver {
public:
    XCSolver();
    virtual ~XCSolver();

    XCSolver(const XCSolver&) = delete;
    XCSolver& operator=(const XCSolver&) = delete;

    virtual double compute_xc(int ngrid, int nao, 
                              const double* d_dm, 
                              const double* d_ao, 
                              const double* d_ao_grad, 
                              const double* d_weights, 
                              double* d_vxc) = 0;

    void compute_coulomb(int nao, const double* d_eri, const double* d_dm, double* d_J);

protected:
    void safe_cublas_dgemm(bool transA, bool transB, int m, int n, int k, 
                           const double* A, int lda, const double* B, int ldb, 
                           double* C, int ldc);
    
    std::unique_ptr<CublasHandleWrapper> handle_wrapper;
};

class LDASolver : public XCSolver {
public:
    LDASolver(); 
    double compute_xc(int ngrid, int nao, 
                      const double* d_dm, 
                      const double* d_ao, 
                      const double* d_ao_grad, 
                      const double* d_weights, 
                      double* d_vxc) override;
};

class GGASolver : public XCSolver {
public:
    GGASolver(); 
    double compute_xc(int ngrid, int nao, 
                      const double* d_dm, 
                      const double* d_ao, 
                      const double* d_ao_grad, 
                      const double* d_weights, 
                      double* d_vxc) override;
};

class B3LYPSolver : public XCSolver {
public:
    B3LYPSolver();
    double compute_xc(int ngrid, int nao, 
                      const double* d_dm, 
                      const double* d_ao, 
                      const double* d_ao_grad, 
                      const double* d_weights, 
                      double* d_vxc) override;
};


extern "C" {
    enum SolverType { 
        SOLVER_LDA = 0, 
        SOLVER_GGA = 1, 
        SOLVER_B3LYP = 2 
    };

    XCSolver* DFT_CreateSolver(int type);
    
    void DFT_DestroySolver(XCSolver* solver);

    double DFT_ComputeXC(XCSolver* solver, int ngrid, int nao,
                         unsigned long long d_dm_ptr,
                         unsigned long long d_ao_ptr,
                         unsigned long long d_ao_grad_ptr,
                         unsigned long long d_weights_ptr,
                         unsigned long long d_vxc_ptr);

    void DFT_ComputeCoulomb(XCSolver* solver, int nao,
                            unsigned long long d_eri_ptr,
                            unsigned long long d_dm_ptr,
                            unsigned long long d_J_ptr);
}