import numpy as np
from pyscf import gto, scf, dft
from scipy.linalg import eigh
# import os, sys, ctypes, numpy as np
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'gansu_bridge.so'))

# # 函数签名与之前一致
# lib.gansu_compute_SH.argtypes = [
#     ctypes.c_char_p, ctypes.c_char_p,
#     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
#     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
#     ctypes.POINTER(ctypes.c_int)
# ]
# lib.gansu_compute_SH.restype = ctypes.c_int


# ==== 1. Build integration grid and AO values ====
def init_grid(mol, grid_add, level=3):
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.prune = None  # 禁用 pruning
    grids.build()
    data = np.loadtxt(grid_add)
    atm_idx_c = data[:, 0].astype(np.int32)
    coords_c = data[:, 1:4]
    weights_c = data[:, 4]
    grids.coords = coords_c
    grids.weights = weights_c
    ao = dft.numint.eval_ao(mol, grids.coords)  # (ngrid, nao)
    print("ao shape: ",ao.shape)# 
    print("grid coords shape:", grids.coords.shape) #every x y z no boxes
    return grids, ao


def init_gridpy(mol, grid_level=3):
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()                              

    ao_values = dft.numint.eval_ao(mol, grids.coords, deriv=0)
    return grids, ao_values

# def compute_SH(atom_str, basis_str):
#     from pyscf import gto
#     nbf = gto.M(atom=atom_str, basis=basis_str, spin=0).nao_nr()
#     S = np.empty((nbf, nbf), dtype=np.double, order='C')
#     H = np.empty((nbf, nbf), dtype=np.double, order='C')
#     ret = lib.gansu_compute_SH(
#         atom_str.encode(),
#         basis_str.encode(),
#         S, H,
#         ctypes.byref(ctypes.c_int(nbf))
#     )
#     if ret != 0:
#         raise RuntimeError("gansu_compute_SH failed")
#     return S, H

def build(atom_structure, grid_add):
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.build()

    
    nao = mol.nao_nr() 
    nelec = mol.nelec[0] + mol.nelec[1]
    nocc = nelec // 2
    
    print(f"基函数数: {nao}")
    print(f"电子数: {nelec}")
    print(f"占据轨道数: {nocc}")

    # grids, ao_values = init_grid(mol, grid_add)
    grids, ao_values = init_gridpy(mol, grid_level=3)
    print(f"积分网格点数: {len(grids.coords)}")
    # S, Hcore = compute_SH(atom_structure, 'sto-3g')
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    Hcore = T + V
    eri = mol.intor('int2e')
    E_nuc = mol.energy_nuc()
    return Hcore, S, nocc, T, eri, ao_values, grids, E_nuc
