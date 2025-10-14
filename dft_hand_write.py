import numpy as np
from pyscf import lib, gto
from pyscf.dft import numint, libxc, gen_grid
import scipy.linalg


molecular_structure = """
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
molecular_structure = """
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.0 0.93 0.0
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
is_converge = False


def get_V_H(dm, eri):

    V_H = np.einsum('uvls,ls->uv', eri, dm)
    return V_H

def get_V_ext(dm, ao_values, grids, xc_code='lda,vwn'):

    nao = ao_values.shape[1]
    ngrids = grids.weights.size
    
    # 计算网格上的电子密度 rho
    rho = numint.eval_rho(mol, ao_values, dm, xctype='LDA')
    
    # 得到 LDA 的 (vrho, vsigma, ...) 数组
    xc = libxc.eval_xc(xc_code, rho, spin=0, deriv=1)   # vrho 是 dE/drho
    vxc = xc[1][0]                                       # (ngrids,)
    
    # 积分 V_xc = Σ_i w_i * ao_i * ao_i^T * vxc_i
    # 这里 ao_values 已经包含网格点值，可直接乘权重
    ao_vxc = ao_values * (grids.weights[:, None] * vxc[:, None])
    V_xc = ao_values.T @ ao_vxc
    
    return V_xc


def solve_fock_equation(F, S):
    """
    求解广义 Fock 方程：F C = S C ε
    输入
        F : (nao, nao) Fock 矩阵
        S : (nao, nao) 重叠矩阵
    返回
        e : (nao,)  分子轨道能级
        C : (nao, nao) 分子轨道系数，按列排列
    """
    # 1. 做对称正交化： X = S^{-1/2}
    # 使用 scipy 的 eigh 保证数值稳定
    s, U = scipy.linalg.eigh(S)             # S 是实对称正定矩阵
    s_sqrt_inv = np.diag(1.0 / np.sqrt(s))
    X = U @ s_sqrt_inv @ U.T                # X = S^{-1/2}

    # 2. 变换到正交基： F' = X^T F X
    F_prime = X.T @ F @ X

    # 3. 对角化 F' 得到本征值和本征向量
    e, C_prime = scipy.linalg.eigh(F_prime)

    # 4. 转回原始 AO 基： C = X C'
    C = X @ C_prime

    return e, C

def compute_E_ext(dm, ao_values, grids, xc_code='lda,vwn'):
    """
    计算交换-相关能 E_xc = ∫ ρ(r) ε_xc(ρ) dr
    dm        : (nao, nao) 密度矩阵
    ao_values : (ngrids, nao) 网格点上的 AO 值
    grids     : 已构建的 Grids 对象
    xc_code   : 泛函字符串
    返回      : 标量 E_xc
    """
    # 网格电子密度 ρ
    rho = numint.eval_rho(mol, ao_values, dm, xctype='LDA')
    
    # 得到能量密度 ε_xc(ρ) 与势 vxc
    exc, vxc = libxc.eval_xc(xc_code, rho, spin=0, deriv=1)[:2]
    
    # 积分： Σ_i w_i * ρ_i * ε_i
    E_xc = np.sum(grids.weights * rho * exc)
    return E_xc

def get_dft_energy(mol):
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    S = mol.intor('int1e_ovlp')
    E_nuc = mol.energy_nuc()
    eri = mol.intor('int2e', aosym='s1')
    nao = mol.nao_nr()
    dm = np.random.rand(nao,nao)*0.01
    ao_values = numint.eval_ao(mol, grids.coords)
    E_old = 0.0
    for iter in range(100):
        V_H = get_V_H(dm, eri)
        V_ext = get_V_ext(dm, ao_values, grids)
        F = H_core + V_H + V_ext

        e, C = solve_fock_equation(F, S)
        dm_new = 2 * C[:,:nocc] @ C[:,:nocc].T

        E_one = np.einsum('ij,ji->', dm_new, H_core)
        E_coul = 0.5*np.einsum('ij,ji->', dm_new, V_H)
        E_exc = compute_E_ext(dm_new, ao_values, grids)
        E_tot = E_one + E_coul + E_exc + E_nuc


        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        print(f"{iter+1:4d}{E_tot:15.8f}{dE:15.8f}{dm_change:15.8f}")
        if abs(dE) < 1e-8 and dm_change < 1e-6:
            is_converge = True
            print("-"*60)
            print(f"SCF converged, E_tot: {E_tot:.8f} Hartree")
            break
        E_old = E_tot
        dm = dm_new

if __name__ == '__main__':

    get_dft_energy(mol)
