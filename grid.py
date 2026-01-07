import numpy as np
from pyscf import gto, scf, dft
from scipy.linalg import eigh


def init_grid(mol, grid_add, level=3):
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.prune = None 
    grids.build()
    data = np.loadtxt(grid_add)
    atm_idx_c = data[:, 0].astype(np.int32)
    coords_c = data[:, 1:4]
    weights_c = data[:, 4]
    grids.coords = coords_c
    grids.weights = weights_c
    ao = dft.numint.eval_ao(mol, grids.coords)  
    print("ao shape: ",ao.shape)
    print("grid coords shape:", grids.coords.shape) 
    return grids, ao


def get_ao_grad(atom_structure, grids):

    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.build()
    ao_all = dft.numint.eval_ao(mol, grids.coords, deriv=1)
    return ao_all[1:4]

def init_gridpy(mol, grid_level=3):
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()                              

    ao_values = dft.numint.eval_ao(mol, grids.coords, deriv=0)
    return grids, ao_values


def build(atom_structure, grid_add):
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.build()

    
    nao = mol.nao_nr() 
    nelec = mol.nelec[0] + mol.nelec[1]
    nocc = nelec // 2
    
    print(f"Number of basis functions: {nao}")
    print(f"Number of electrons: {nelec}")
    print(f"Number of occupied orbitals: {nocc}")

    # grids, ao_values = init_grid(mol, grid_add)
    grids, ao_values = init_gridpy(mol, grid_level=3)
    print(f"Number of grid points for integration: {len(grids.coords)}")
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    Hcore = T + V
    eri = mol.intor('int2e')
    E_nuc = mol.energy_nuc()
    return Hcore, S, nocc, T, eri, ao_values, grids, E_nuc
