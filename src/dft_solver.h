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
    double compute_xc(int ngrid, int nao, 
                      const double* d_dm, 
                      const double* d_ao, 
                      const double* d_ao_grad, 
                      const double* d_weights, 
                      double* d_vxc) override;
};