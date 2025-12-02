
import numpy as np
from scipy.linalg import eigh
import sys, ctypes, os, time
from datetime import timedelta
from pyscf import gto, dft
from grid import build, get_ao_grad

libname = {'linux':'./weights/gga.so',
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
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # grad_rho (NEW)
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
lib.build_vxc_matrix_gga.restype = None

# -------- 2. compute_exc_energy_gga ----------
lib.compute_exc_energy_gga.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # weights
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # rho
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # sigma
lib.compute_exc_energy_gga.restype = ctypes.c_double

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

lib.get_rho_sigma.restype = None
lib.get_rho_sigma.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # dm
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # ao
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # ao_grad
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # rho
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # sigma
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),  # grad_rho (NEW)
]



def get_rho_grad(dm, ao_values, ao_grad):

    ngrid, nao = ao_values.shape
    rho   = np.empty(ngrid, dtype=np.float64)
    sigma = np.empty(ngrid, dtype=np.float64)
    grad_rho = np.empty((ngrid, 3), dtype=np.float64) # NEW

    lib.get_rho_sigma(nao, ngrid,
                      np.ascontiguousarray(dm, dtype=np.float64),
                      np.ascontiguousarray(ao_values, dtype=np.float64),
                      np.ascontiguousarray(ao_grad,  dtype=np.float64),
                      rho, sigma, grad_rho) # Pass NEW arg
    
    rho   = np.clip(rho,   1e-12, None)
    # sigma = np.clip(sigma, 0.0, 1e4)
    return rho, sigma, grad_rho



def build_vxc_matrix(dm, ao_values, ao_grad, grids):
    rho, sigma, grad_rho = get_rho_grad(dm, ao_values, ao_grad)
    nao, ngrid = ao_values.shape[1], ao_values.shape[0]

    ao_c   = np.ascontiguousarray(ao_values,  dtype=np.float64)
    aograd_c = np.ascontiguousarray(ao_grad, dtype=np.float64)
    w_c    = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c  = np.ascontiguousarray(rho,  dtype=np.float64)
    sig_c  = np.ascontiguousarray(sigma, dtype=np.float64)
    grho_c = np.ascontiguousarray(grad_rho, dtype=np.float64) # NEW
    vxc_mat = np.empty((nao, nao), dtype=np.float64, order='C')

    lib.build_vxc_matrix_gga(nao, ngrid, ao_c, aograd_c, w_c, rho_c, sig_c, grho_c, vxc_mat)
    return vxc_mat

def compute_exc_energy(dm, ao_values, ao_grad, grids):
    rho, sigma, _ = get_rho_grad(dm, ao_values, ao_grad) # Ignore grad_rho here
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
    if cycle < 10:
        mix_param = 0.1
    elif dm_change > 1e-3:
        mix_param = 0.2
    elif dm_change > 1e-4:
        mix_param = 0.3
    else:
        mix_param = 0.5
    return mix_param * dm_new + (1 - mix_param) * dm_old



if __name__ == "__main__":
    atom = "h2o"
    with open(f"./atom_txt/{atom}.txt", "r") as f:
        atom_structure = f.read()
    grid_add = f"./grid_txt/{atom}_grid.txt"
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_add)
    ao_grad = get_ao_grad(atom_structure, grids)

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
            print(f' E_one  : {E_one:.6f} Hartree')
            print(f' E_coul : {E_coul:.6f} Hartree')
            print(f' E_exc  : {E_xc:.6f} Hartree')
            print(f"Vxc_average_time: {sum(Vxc_time)/len(Vxc_time)*1000:.6f} ms")
            print(f"Exc_average_time: {sum(Exc_time)/len(Exc_time)*1000:.6f} ms")
            print(f"Total time: {end_time - start_time:.6f} s\n")
            break

        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("Warning: SCF unconverged!")

    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()

    start = time.time()
    mf = dft.RKS(mol)
    mf.xc = 'PBE'       
    mf.kernel()
    dm = mf.make_rdm1()

    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    E1 = np.einsum('ij,ji->', h1, dm)
    vh = mf.get_j(mol, dm)
    Ecoul = 0.5 * np.einsum('ij,ji->', vh, dm)
    Exc   = mf.energy_elec()[0] - (E1 + Ecoul)
    Etot  = mf.energy_tot()
    elapsed = time.time() - start

    print('PySCF (PBE) reference:')
    print(f' E_one  : {E1:.6f} Hartree')
    print(f' E_coul : {Ecoul:.6f} Hartree')
    print(f' E_exc  : {Exc:.6f} Hartree')
    print(f' E_tot  : {Etot:.8f} Hartree')
    print(f' time   : {elapsed:.6f} s')


