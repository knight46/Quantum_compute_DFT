import numpy as np
from scipy.linalg import eigh
import sys
import ctypes, numpy as np, os, sys
from grid import build
from pyscf import gto, dft
import time
from datetime import timedelta


libname = {'linux':'./weights/lda.so',
           'darwin':'lda.so',
           'win32':'dft.dll'}[sys.platform]
lib = ctypes.CDLL(os.path.abspath(libname))

# ---------- 1. lda_exc_vxc ----------
# lib.lda_exc_vxc.argtypes = [
#     ctypes.c_int,
#     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
#     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
#     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
# ]
# lib.lda_exc_vxc.restype = None

# ---------- 2. build_vxc_matrix ----------
lib.build_vxc_matrix.argtypes = [
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
]
lib.build_vxc_matrix.restype = None

# ---------- 3. compute_exc_energy ----------
lib.compute_exc_energy.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
]
lib.compute_exc_energy.restype = ctypes.c_double

# ---------- 4. build_coulomb_matrix ----------
lib.build_coulomb_matrix.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
]
lib.build_coulomb_matrix.restype = None

lib.solve_fock_eigen.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
]
lib.solve_fock_eigen.restype = None

lib.get_rho.argtypes = [
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
]
lib.get_rho.restype = None



# ==== 2. Build LDA exchange-correlation functional ====


def lda_exc_vxc(rho):
    rho = np.asarray(rho, dtype=np.float64, order='C')
    n = rho.size
    exc = np.empty_like(rho)
    vxc = np.empty_like(rho)
    lib.lda_exc_vxc(n, rho, exc, vxc)
    return exc, vxc



# ==== 3. Compute density on grid ====

def get_rho(dm, ao_values):
    nao, ngrid = ao_values.shape[1], ao_values.shape[0]
    rho = np.empty(ngrid, dtype=np.float64)
    lib.get_rho(nao, ngrid,
                np.ascontiguousarray(dm, dtype=np.float64),
                np.ascontiguousarray(ao_values, dtype=np.float64),
                rho)
    return rho


# ==== 4. Build Vxc matrix ====
def build_vxc_matrix(dm, ao_values, grids):
    rho = get_rho(dm, ao_values)
    nao   = ao_values.shape[1]
    ngrid = ao_values.shape[0]

    ao_c  = np.ascontiguousarray(ao_values,  dtype=np.float64)
    w_c   = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c = np.ascontiguousarray(rho, dtype=np.float64)
    vxc_mat = np.empty((nao, nao), dtype=np.float64, order='C')
    # print(f"nao:{nao}, ngrid:{ngrid},ao_c:{ao_c.shape}, w_c:{w_c.shape}, rho_c:{rho_c.shape}, vxc_mat:{vxc_mat.shape}")
    # test = input("t")
    lib.build_vxc_matrix(nao, ngrid, ao_c, w_c, rho_c, vxc_mat)
    return vxc_mat


# ==== 5. Compute Exc energy ====
def compute_exc_energy(dm, ao_values, grids):
    rho = get_rho(dm, ao_values)
    rho_c = np.ascontiguousarray(rho, dtype=np.float64)
    w_c   = np.ascontiguousarray(grids.weights, dtype=np.float64)
    return lib.compute_exc_energy(len(grids.coords), w_c, rho_c)

def build_coulomb_matrix(dm, eri):
    nao = dm.shape[0]
    dm_c = np.ascontiguousarray(dm,  dtype=np.float64)
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


# def LDA(mol):

#     mf = dft.RKS(mol)
#     mf.xc = 'LDA,VWN'  
#     energy = mf.kernel()
#     return mf

def adaptive_mixing(dm_new, dm_old, cycle, dm_change):
    """
    自适应密度混合
    """
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
    # ==== 0. Molecule Definition ====
    atom = "h2o"
    with open(f"./atom_txt/{atom}.txt", "r") as f:
        atom_structure = f.read()
 
    grid_add = f"./grid_txt/{atom}_grid.txt"
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_add)
    e_init, C_init = eigh(Hcore, S)
    dm = 2 * C_init[:, :nocc] @ C_init[:, :nocc].T
    start_time = time.time()
    
    print(f"\nSCF started!")
    print("-" * 70)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12}")
    print("-" * 70)
    
    converged = False
    E_old = 0.0
    Vxc_time = []
    Exc_time = []

    for cycle in range(100):
        J = build_coulomb_matrix(dm, eri)
        # break
        Vxc_start = time.time()
        Vxc = build_vxc_matrix(dm, ao_values, grids)
        Vxc_end = time.time()
        Vxc_time.append(Vxc_end - Vxc_start)
        end = time.time()
        # break
        F = Hcore + J + Vxc
        e, C = solve_fock_equation(F, S)
        # break
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T

        E_one = np.einsum('ij,ji->', dm_new, Hcore)
        E_coul = 0.5 * np.einsum('ij,ji->', dm_new, J)
        Exc_start = time.time()
        E_xc = compute_exc_energy(dm_new, ao_values, grids)
        Exc_end = time.time()
        Exc_time.append(Exc_end - Exc_start)
        E_tot = E_one + E_coul + E_xc + E_nuc
        
        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)

        print(f"{cycle+1:4d} {E_tot:15.8f} {dE:12.6e} {dm_change:12.6e}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            # print("-" * 60)
            # print(f"E_one:{E_one:.8f} Hartree")
            # print(f"E_coul:{E_coul:.8f} Hartree")
            # print(f"E_xc:{E_xc:.8f} Hartree")
            # print(f"E_nuc:{E_nuc:.8f} Hartree")
            end_time = time.time()
            print(f"SCF converged! E = {E_tot:.8f} Hartree")
            print(f"Vxc_average_time: {(sum(Vxc_time)/len(Vxc_time)*1000):.6f} ms")
            print(f"Exc_average_time: {(sum(Exc_time)/len(Exc_time)*1000):.6f} ms")
            print(f"expense total time: {(end_time - start_time):.6f} s\n")
            break
        
        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("warning: SCF unconverged!")

    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()
    start_time = time.time()
    def LDA(mol):
        mf = dft.RKS(mol)
        mf.xc = 'LDA,VWN'  
        energy = mf.kernel()
        return mf
    mf = LDA(mol)
    dm = mf.make_rdm1()

    # 一电子项（动能 + 核吸引）
    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    E_one = np.einsum('ij,ji->', h1, dm)

    # Hartree + XC 势
    veff = mf.get_veff(mol, dm)

    # Coulomb 能量 (0.5 * ρV_H)
    vh = mf.get_j(mol, dm)
    E_coul = 0.5 * np.einsum('ij,ji->', vh, dm)

    # 交换-相关能量
    E_exc = mf.energy_elec()[0] - (E_one + E_coul)

    # 总能量
    E_tot = mf.energy_tot()
    end_time = time.time()
    cost_time = end_time - start_time
    print(f' E_one : {E_one:.6f} Hartree')
    print(f' E_coul : {E_coul:.6f} Hartree')
    print(f' E_exc : {E_exc:.6f} Hartree')
    print(f' E_tot : {mf.energy_tot():.8f} Hartree')
    print(f' time_cost: {cost_time:.6f} s')