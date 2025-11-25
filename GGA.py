# gga.py  – 与 gga.cu 配套的驱动脚本
import numpy as np
from scipy.linalg import eigh
import sys, ctypes, os, time
from datetime import timedelta
from pyscf import gto, dft
from grid import build, get_ao_grad

# ---------------- 载入 CUDA 共享库 -----------------
libname = {'linux':'mxgga.so',
           'darwin':'libgga.so',
           'win32':'dft.dll'}[sys.platform]
lib = ctypes.CDLL(os.path.abspath(libname))

# -------- 1. build_vxc_matrix_gga ----------
lib.build_vxc_matrix_gga.argtypes = [
    ctypes.c_int,                       # nao
    ctypes.c_int,                       # ngrid
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # ao
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # ao_grad
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # weights
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # rho
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # sigma
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # vxc_mat
lib.build_vxc_matrix_gga.restype = None

# -------- 2. compute_exc_energy_gga ----------
lib.compute_exc_energy_gga.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # weights
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # rho
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # sigma
lib.compute_exc_energy_gga.restype = ctypes.c_double

# -------- 3. 其余接口与 dft_cuda.cpp 相同 ----------
lib.build_coulomb_matrix.argtypes = [ctypes.c_int,
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
lib.build_coulomb_matrix.restype = None

lib.solve_fock_eigen.argtypes = [ctypes.c_int,
                                 np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
lib.solve_fock_eigen.restype = None

lib.get_rho.argtypes = [ctypes.c_int, ctypes.c_int,
                        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
lib.get_rho.restype = None

# ===================================================
#  工具函数
# ===================================================
def get_rho_grad(dm, ao_values, ao_grad):
    """
    返回 rho 与 sigma = |∇ρ|²
    ao_grad: (ngrid, 3, nao)
    """
    ngrid, nao = ao_values.shape[0], ao_values.shape[1]
    rho = np.empty(ngrid, dtype=np.float64)
    lib.get_rho(nao, ngrid,
                np.ascontiguousarray(dm, dtype=np.float64),
                np.ascontiguousarray(ao_values, dtype=np.float64),
                rho)

    # 计算 ∇ρ
    grad_rho = np.einsum('gmi,ij,gj->gm', ao_grad, dm, ao_values) * 2.0
    sigma = np.einsum('gm,gm->g', grad_rho, grad_rho)
    sigma = np.abs(sigma)                # 强制非负
    sigma = np.clip(sigma, 0.0, None)    # 二次保险
    if sigma.min() < 0:
        print('[WARN] sigma < 0 detected!', sigma.min())
    return rho, sigma


def build_vxc_matrix(dm, ao_values, ao_grad, grids):
    rho, sigma = get_rho_grad(dm, ao_values, ao_grad)
    nao, ngrid = ao_values.shape[1], ao_values.shape[0]

    ao_c   = np.ascontiguousarray(ao_values,  dtype=np.float64)
    aograd_c = np.ascontiguousarray(ao_grad.reshape(ngrid, 3*nao), dtype=np.float64)
    w_c    = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c  = np.ascontiguousarray(rho,  dtype=np.float64)
    sig_c  = np.ascontiguousarray(sigma, dtype=np.float64)
    vxc_mat = np.empty((nao, nao), dtype=np.float64, order='C')

    lib.build_vxc_matrix_gga(nao, ngrid, ao_c, aograd_c, w_c, rho_c, sig_c, vxc_mat)
    return vxc_mat


def compute_exc_energy(dm, ao_values, ao_grad, grids):
    rho, sigma = get_rho_grad(dm, ao_values, ao_grad)
    w_c   = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c = np.ascontiguousarray(rho,  dtype=np.float64)
    sig_c = np.ascontiguousarray(sigma, dtype=np.float64)
    return lib.compute_exc_energy_gga(len(grids.coords), w_c, rho_c, sig_c)


def build_coulomb_matrix(dm, eri):
    nao = dm.shape[0]
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    eri_c = np.ascontiguousarray(eri.reshape(-1), dtype=np.float64)
    J = np.empty((nao, nao), dtype=np.float64, order='C')
    lib.build_coulomb_matrix(nao, eri_c, dm_c, J)
    return J


def solve_fock_equation(F, S):
    n = F.shape[0]
    e = np.empty(n)
    C = np.empty((n, n), order='C')
    lib.solve_fock_eigen(n,
                         np.ascontiguousarray(F, dtype=np.float64),
                         np.ascontiguousarray(S, dtype=np.float64),
                         e, C)
    C = C.reshape(n, n).T
    return e, C


def adaptive_mixing(dm_new, dm_old, cycle, dm_change):
    # if cycle < 10:
    #     mix_param = 0.1
    # elif dm_change > 1e-3:
    #     mix_param = 0.2
    # elif dm_change > 1e-4:
    #     mix_param = 0.3
    # else:
    #     mix_param = 0.5
    # return mix_param * dm_new + (1 - mix_param) * dm_old
    return 0.05 * dm_new + 0.95 * dm_old   # 超保守混合


# ===================================================
#  主程序
# ===================================================
if __name__ == "__main__":
    # ---- 0. 读入分子与网格 ----
    with open("./atom_txt/c4h10.txt", "r") as f:
        atom_structure = f.read()
    grid_add = "./grid_txt/C4H10_grid.txt"
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_add)
    # Hcore, S, nocc, T, eri, ao_values, grids, E_nuc
    ao_grad = get_ao_grad(atom_structure, grids)
    # ---- 初始猜测 ----
    e_init, C_init = eigh(Hcore, S)
    dm = 2 * C_init[:, :nocc] @ C_init[:, :nocc].T
    start_time = time.time()

    print("\nSCF started!")
    print("-" * 70)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12}")
    print("-" * 70)

    converged = False
    E_old = 0.0
    Vxc_time, Exc_time = [], []

    for cycle in range(100):
        J = build_coulomb_matrix(dm, eri)

        t1 = time.time()
        Vxc = build_vxc_matrix(dm, ao_values, ao_grad, grids)
        t2 = time.time()
        Vxc_time.append(t2 - t1)

        F = Hcore + J + Vxc
        e, C = solve_fock_equation(F, S)
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T

        E_one  = np.einsum('ij,ji->', dm_new, Hcore)
        E_coul = 0.5 * np.einsum('ij,ji->', dm_new, J)

        t3 = time.time()
        E_xc = compute_exc_energy(dm_new, ao_values, ao_grad, grids)
        t4 = time.time()
        Exc_time.append(t4 - t3)

        E_tot = E_one + E_coul + E_xc + E_nuc
        dE    = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)

        print(f"{cycle+1:4d} {E_tot:15.8f} {dE:12.6e} {dm_change:12.6e}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            end_time = time.time()
            print(f"SCF converged! E = {E_tot:.8f} Hartree")
            print(f"Vxc_average_time: {sum(Vxc_time)/len(Vxc_time)*1000:.6f} ms")
            print(f"Exc_average_time: {sum(Exc_time)/len(Exc_time)*1000:.6f} ms")
            print(f"Total time: {end_time - start_time:.6f} s")
            break

        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("Warning: SCF unconverged!")

    # ---------------- PySCF 对照组（GGA-PBE） ----------------
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()

    start = time.time()
    mf = dft.RKS(mol)
    mf.xc = 'PBE'          # ← GGA
    mf.kernel()
    dm = mf.make_rdm1()

    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    E1 = np.einsum('ij,ji->', h1, dm)
    vh = mf.get_j(mol, dm)
    Ecoul = 0.5 * np.einsum('ij,ji->', vh, dm)
    Exc   = mf.energy_elec()[0] - (E1 + Ecoul)
    Etot  = mf.energy_tot()
    elapsed = time.time() - start

    print('\nPySCF (PBE) reference:')
    print(f' E_one  : {E1:.6f} Hartree')
    print(f' E_coul : {Ecoul:.6f} Hartree')
    print(f' E_exc  : {Exc:.6f} Hartree')
    print(f' E_tot  : {Etot:.8f} Hartree')
    print(f' time   : {elapsed:.6f} s')