import numpy as np
from pyscf import lib, gto
from pyscf.dft import numint, libxc, gen_grid
import scipy.linalg

# 体系：H2 分子
molecular_structure = """
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""

mol = gto.Mole()
mol.atom = molecular_structure
mol.basis = 'sto-6g'
mol.unit = 'Ang'
mol.build()
nelec = mol.nelectron
nocc = nelec // 2

grids = gen_grid.Grids(mol)
grids.build()

# ----------------- 工具函数 -----------------
def get_V_H(dm, eri):
    """Hartree 势矩阵"""
    return np.einsum('uvls,ls->uv', eri, dm)

def get_V_xc_GGA(dm, ao_values, grids, xc_code='pbe,pbe'):
    """
    GGA 交换-相关势矩阵
    ao_values : (ngrids, nao) 已含一阶导数（由 deriv=1 产生）
    """
    # rho 形状为 (4, ngrids)：rho, grad_x, grad_y, grad_z
    rho = numint.eval_rho(mol, ao_values, dm, xctype='GGA')
    exc, vxc = libxc.eval_xc(xc_code, rho, spin=0, deriv=1)[:2]
    # numint.eval_mat 把 vxc 组装成 V_xc 矩阵
    V_xc = numint.eval_mat(mol, ao_values, grids.weights, rho, vxc, xctype='GGA')
    return V_xc

def solve_fock_equation(F, S):
    """广义本征方程求解"""
    s, U = scipy.linalg.eigh(S)
    X = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
    F_prime = X.T @ F @ X
    e, C_prime = scipy.linalg.eigh(F_prime)
    C = X @ C_prime
    return e, C

def compute_E_xc_GGA(dm, ao_values, grids, xc_code='pbe,pbe'):
    """GGA 交换-相关能"""
    rho = numint.eval_rho(mol, ao_values, dm, xctype='GGA')
    exc, vxc = libxc.eval_xc(xc_code, rho, spin=0, deriv=1)[:2]
    return np.einsum('i,i,i->', grids.weights, rho[0], exc)

# ----------------- SCF 主循环 -----------------
def get_dft_energy(mol):
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    S = mol.intor('int1e_ovlp')
    E_nuc = mol.energy_nuc()
    eri = mol.intor('int2e', aosym='s1')
    nao = mol.nao_nr()

    # 初猜密度
    dm = np.random.rand(nao, nao) * 0.01
    ao_values = numint.eval_ao(mol, grids.coords, deriv=1)  # 需要导数做 GGA
    E_old = 0.0
    for it in range(100):
        V_H = get_V_H(dm, eri)
        V_xc = get_V_xc_GGA(dm, ao_values, grids)
        F = H_core + V_H + V_xc

        e, C = solve_fock_equation(F, S)
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T

        E_one = np.einsum('ij,ji->', dm_new, H_core)
        E_coul = 0.5 * np.einsum('ij,ji->', dm_new, V_H)
        E_xc = compute_E_xc_GGA(dm_new, ao_values, grids)
        E_tot = E_one + E_coul + E_xc + E_nuc

        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        print(f"{it+1:4d}{E_tot:18.10f}{dE:14.8f}{dm_change:14.8f}")
        if abs(dE) < 1e-8 and dm_change < 1e-6:
            print("-" * 60)
            print(f"SCF converged, E_tot = {E_tot:.10f} Hartree")
            break
        E_old = E_tot
        dm = dm_new
    else:
        print("SCF not converged in 100 iterations")

if __name__ == '__main__':
    get_dft_energy(mol)