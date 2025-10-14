import numpy as np
from pyscf import lib, gto, dft
from pyscf.dft import numint, libxc, gen_grid
from pyscf.scf import hf
import scipy.linalg

# molecular_structure = """
# C   0.000000  0.000000  0.000000
# H   0.629118  0.629118  0.629118
# H  -0.629118 -0.629118  0.629118
# H  -0.629118  0.629118 -0.629118
# H   0.629118 -0.629118 -0.629118
# """

# molecular_structure = """
# O 0.0 0.0 0.0
# H 0.0 0.0 0.96
# H 0.0 0.93 0.0
# """


molecular_structure = 'H 0.0 0.0 0.0; H 0.0 0.0 0.74'  

mol = gto.Mole()
mol.atom = molecular_structure
mol.basis = 'sto-6g'
mol.unit = 'Ang'
mol.build()
nelec = mol.nelectron 
nocc = nelec // 2 
grids = gen_grid.Grids(mol)
grids.level = 3      
grids.build()

def get_V_H(dm, eri):
    return np.einsum('uvls,ls->uv', eri, dm)

def get_V_x_HF(dm, eri):
    V_x_HF = -np.einsum('uvls,sl->uv', eri, dm)
    return V_x_HF

def solve_fock_equation(F, S):
    s, U = scipy.linalg.eigh(S)
    s_sqrt_inv = np.diag(1.0 / np.sqrt(s))
    X = U @ s_sqrt_inv @ U.T
    F_prime = X.T @ F @ X
    e, C_prime = scipy.linalg.eigh(F_prime)
    C = X @ C_prime
    return e, C

def get_E_x_HF(dm, V_x_HF):
    E_x_HF = 0.5 * np.einsum('uv,uv->', dm, V_x_HF)
    return E_x_HF

def get_dft_energy(mol):
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    S = mol.intor('int1e_ovlp')
    E_nuc = mol.energy_nuc()
    eri = mol.intor('int2e', aosym='s1')
    nao = mol.nao_nr()
    dm = np.random.rand(nao, nao) * 0.01
    E_old = 0.0
    damping = 0.3
    ni = numint.NumInt()


    for iter in range(100):
        V_H = get_V_H(dm, eri)
        V_x_HF = get_V_x_HF(dm, eri)
        E_x_HF = get_E_x_HF(dm, V_x_HF)
        xc_code = '0.08*LDA_X + 0.72*B88 + 0.20*HF, 0.19*VWN5 + 0.81*LYP'
        _, E_xc, V_xc = ni.nr_vxc(mol, grids, xc_code, (dm,), spin=0)
        F = H_core + V_H + V_xc 

        e, C = solve_fock_equation(F, S)
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T 

        E_one = np.einsum('ij,ji->', dm_new, H_core)
        E_coul = 0.5 * np.einsum('ij,ji->', dm_new, V_H)
        E_tot = E_one + E_coul + E_xc + E_nuc 
        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        print(f"{iter+1:4d}{E_tot:15.8f}{dE:15.8f}{dm_change:15.8f}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            print("-" * 60)
            print(f"E_one : {E_one:.8f} Hartree")
            print(f"E_coul: {E_coul:.8f} Hartree")
            print(f"E_xc  : {E_xc:.8f} Hartree")
            print(f"E_nuc : {E_nuc:.8f} Hartree")
            # print(f"E_x_HF: {c_hf * E_x_HF:.8f} Hartree")
            print(f"SCF converged, E_tot: {E_tot:.8f} Hartree")
            break
        E_old = E_tot
        # dm = dm_new
        dm = (1 - damping) * dm + damping * dm_new

if __name__ == '__main__':
    get_dft_energy(mol)