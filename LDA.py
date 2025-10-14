import numpy as np
from scipy.linalg import eigh
import sys
import ctypes, numpy as np, os, sys
from grid import build
from pyscf import gto, dft
import time
from datetime import timedelta


libname = {'linux':'lda.so',
           'darwin':'lda.so',
           'win32':'dft.dll'}[sys.platform]
lib = ctypes.CDLL(os.path.abspath(libname))



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
    if cycle < 5:
        mix_param = 0.3
    elif dm_change > 1e-3:
        mix_param = 0.2
    elif dm_change > 1e-4:
        mix_param = 0.5
    else:
        mix_param = 0.7
    
    return mix_param * dm_new + (1 - mix_param) * dm_old

if __name__ == "__main__": 
    # ==== 0. Molecule Definition ====
    # atom_structure = 'H 0.0 0.0 0.0; H 0.0 0.0 0.74'
    
    # atom_structure = '''
    #     C	-0.429	0.643	0.000
    #     C	0.429	-0.643	0.000
    #     C	0.429	1.923	0.000
    #     C	-0.429	-1.923	0.000
    #     H	-1.075	0.643	0.876
    #     H	-1.075	0.643	-0.876
    #     H	1.075	-0.643	0.876
    #     H	1.075	-0.643	-0.876
    #     H	-0.203	2.807	0.000
    #     H	1.065	1.960	0.880
    #     H	1.065	1.960	-0.880
    #     H	0.203	-2.807	0.000
    #     H	-1.065	-1.960	0.880
    #     H	-1.065	-1.960	-0.880
    # ''' 
    atom_structure ="""
        C   0.000    1.387    0.000
        C   1.201    0.693    0.000
        C   1.201   -0.693    0.000
        C   0.000   -1.387    0.000
        C  -1.201   -0.693    0.000
        C   -1.201    0.693    0.000
        H   0.000    2.469    0.000 
        H   2.139    1.235    0.000
        H   2.139   -1.235    0.000
        H   0.000   -2.469   0.000
        H  -2.139   -1.235    0.000
        H  -2.139    1.235    0.000
    """
    # atom_structure = 'O 0.0 0.0 0.0; H 0.0 0.0 0.96; H 0.0 0.93 0.0'

    # atom_structure ="""
    #     O	-3.7479	-2.7792	0.4882
    #     O	-2.3738	-3.0111	-1.3071
    #     C	-2.513	-0.1909	-1.8131
    #     C	-1.1764	2.5209	-0.9409
    #     C	-2.0031	1.5018	1.758
    #     C	-0.8804	-1.2824	2.021
    #     C	-3.5804	-0.937	-1.0169
    #     C	-2.9526	1.1955	-2.1916
    #     C	-2.3833	2.352	-1.8174
    #     C	1.8815	-2.3594	-0.6125
    #     C	-1.4629	3.4123	0.231
    #     C	-1.8326	2.9576	1.438
    #     C	-1.0853	1.0718	2.8652
    #     C	4.459	-0.5303	-0.9467
    #     C	-0.5866	-0.1682	2.9803
    #     C	0.1014	-1.3198	0.8874
    #     C	0.9023	-2.3337	0.5245
    #     C	3.2757	-2.6377	-0.1273
    #     C	4.3658	-1.869	-0.2753
    #     C	4.8875	0.5356	0.0198
    #     C	2.8016	1.8907	-0.2761
    #     C	4.1498	1.6148	0.3194
    #     C	-3.1614	-2.3402	-0.6531
    #     C	2.9151	2.7057	-1.5545
    #     H	-2.311	-0.737	-2.7438
    #     H	-1.5693	-0.1983	-1.2671
    #     H	-0.4105	3.0246	-1.5444
    #     H	-0.7038	1.5848	-0.6439
    #     H	-3.0337	1.3765	2.1153
    #     H	-1.9147	0.8661	0.8812
    #     H	-1.8963	-1.201	1.6271
    #     H	-0.859	-2.2281	2.5767
    #     H	-4.5133	-1.0083	-1.5879
    #     H	-3.7928	-0.3882	-0.0919
    #     H	-3.8205	1.2536	-2.846
    #     H	-2.826	3.2695	-2.2031
    #     H	1.5903	-3.1629	-1.2997
    #     H	1.8169	-1.4307	-1.1855
    #     H	-1.3824	4.4857	0.0789
    #     H	-2.0384	3.6789	2.2252
    #     H	-0.8358	1.795	3.6385
    #     H	3.5528	-0.2821	-1.4992
    #     H	5.2385	-0.6044	-1.7166
    #     H	0.071	-0.3905	3.8169
    #     H	0.1702	-0.4088	0.2972
    #     H	0.8333	-3.2545	1.1027
    #     H	3.4136	-3.5908	0.3825
    #     H	5.2993	-2.2571	0.1313
    #     H	5.8628	0.4282	0.4887
    #     H	2.2272	0.9773	-0.4536
    #     H	2.2161	2.4583	0.458
    #     H	4.5497	2.3472	1.0155
    #     H	1.9216	2.9259	-1.9551
    #     H	3.4225	3.6591	-1.3723
    #     H	3.474	2.1652	-2.3257
    #     H	-3.4626	-3.6887	0.7199
    # """

    grid_add = "./grid_txt/C6H6_grid.txt"
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
    # print(f"dm:\n{dm}\n")
    # print(f"eri:\n{eri}\n")
    # print(f"ao_values:\n{ao_values}\n")

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
            print(f"expense total time: {(end_time - start_time):.6f} s")
            break
        
        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("warning: SCF unconverged!")

    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()
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

    print(f' E_one : {E_one:.6f} Hartree')
    print(f' E_coul : {E_coul:.6f} Hartree')
    print(f' E_exc : {E_exc:.6f} Hartree')
    print(f' E_tot : {mf.energy_tot():.8f} Hartree')

