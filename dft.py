import numpy as np
import cupy as cp
from scipy.linalg import eigh
import sys, ctypes, os, time
from pyscf import gto, dft
from pyscf.scf import diis
from grid import build, get_ao_grad
import argparse


class DFTSolverWrapper:

    TYPE_LDA = 0
    TYPE_GGA = 1

    def __init__(self, lib_path, functional_type='lda'):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found at: {lib_path}")
        
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        self.functional_type = functional_type.upper()
        
        self.lib.DFT_CreateSolver.argtypes = [ctypes.c_int]
        self.lib.DFT_CreateSolver.restype = ctypes.c_void_p

        self.lib.DFT_DestroySolver.argtypes = [ctypes.c_void_p]
        self.lib.DFT_DestroySolver.restype = None

        self.lib.DFT_ComputeXC.argtypes = [
            ctypes.c_void_p, 
            ctypes.c_int, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
        ]
        self.lib.DFT_ComputeXC.restype = ctypes.c_double

        self.lib.DFT_ComputeCoulomb.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
        ]

        c_type = self.TYPE_GGA if self.functional_type == 'GGA' else self.TYPE_LDA
        self.solver = self.lib.DFT_CreateSolver(c_type)
        if not self.solver:
            raise RuntimeError("Failed to create C++ DFT Solver instance.")

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'solver') and self.solver:
            self.lib.DFT_DestroySolver(self.solver)

    def compute_xc(self, ngrid, nao, d_dm, d_ao, d_weights, d_vxc, d_ao_grad=None):

        ptr_dm = d_dm.data.ptr
        ptr_ao = d_ao.data.ptr
        ptr_w  = d_weights.data.ptr
        ptr_vxc = d_vxc.data.ptr
        
        ptr_grad = d_ao_grad.data.ptr if d_ao_grad is not None else 0

        energy = self.lib.DFT_ComputeXC(
            self.solver,
            ngrid, nao,
            ctypes.c_uint64(ptr_dm),
            ctypes.c_uint64(ptr_ao),
            ctypes.c_uint64(ptr_grad),
            ctypes.c_uint64(ptr_w),
            ctypes.c_uint64(ptr_vxc)
        )
        return energy

    def compute_coulomb(self, nao, d_eri, d_dm, d_J):

        self.lib.DFT_ComputeCoulomb(
            self.solver,
            nao,
            ctypes.c_uint64(d_eri.data.ptr),
            ctypes.c_uint64(d_dm.data.ptr),
            ctypes.c_uint64(d_J.data.ptr)
        )



def load_xyz_as_pyscf_atom(xyz_path):
    with open(xyz_path, "r") as f: lines = f.readlines()
    return "".join(lines[2:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DFT (LDA/GGA) using CUDA backend.")
    parser.add_argument("functional", type=str, choices=["LDA", "GGA"], help="Functional type (LDA or GGA)")
    parser.add_argument("xyzfile", type=str, help="Molecule name (e.g., H2O)")
    args = parser.parse_args()

    lib_path = "./weights/dft.so"
    atom_file = args.xyzfile if args.xyzfile.lower().endswith(".xyz") else args.xyzfile + ".xyz"
    grid_path = f"./grid_txt/{atom_file.replace('.xyz', '')}_grid.txt"
    atom_path = f"./atom_txt/{atom_file}"

    if not os.path.exists(atom_path):
        print(f"Error: {atom_path} not found.")
        exit(1)

    print(f"=== DFT Solver: {args.functional} | Molecule: {atom_file} ===")


    print("Building CPU data...")
    atom_structure = load_xyz_as_pyscf_atom(atom_path)
    

    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_path)
    
    ngrid, nao = ao_values.shape
    print(f"System Info: NAO={nao}, Grid={ngrid}, Occupied={nocc}")

    ao_grad = None
    if args.functional == "GGA":
        print("Calculating AO Gradients (GGA mode)...")
        ao_grad = get_ao_grad(atom_structure, grids)
    else:
        print("Skipping AO Gradients (LDA mode).")

    print("Moving data to GPU...")
    t_start_gpu = time.time()

    try:
        dft_solver = DFTSolverWrapper(lib_path, args.functional)
    except Exception as e:
        print(e)
        exit(1)

    d_ao = cp.asarray(ao_values, dtype=np.float64, order='C')
    d_weights = cp.asarray(grids.weights, dtype=np.float64, order='C')
    d_eri = cp.asarray(eri.reshape(nao*nao, nao*nao), dtype=np.float64, order='C')
    
    d_ao_grad = None
    if ao_grad is not None:
        d_ao_grad = cp.asarray(ao_grad, dtype=np.float64, order='C')

    d_dm  = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_J   = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_vxc = cp.zeros((nao, nao), dtype=np.float64, order='C')

    cp.cuda.Stream.null.synchronize()
    print(f"GPU Init Time: {time.time() - t_start_gpu:.4f}s")

    e, C = eigh(Hcore, S)
    dm = 2 * C[:, :nocc] @ C[:, :nocc].T
    
    adiis = diis.CDIIS()

    print("\nSCF started!")
    print("-" * 65)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12}")
    print("-" * 65)

    E_old = 0.0
    converged = False
    start_time = time.time()
    
    xc_times = [] 

    for cycle in range(200):
        d_dm.set(dm)

        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        dft_solver.compute_coulomb(nao, d_eri, d_dm, d_J)
        cp.cuda.Stream.null.synchronize()


        t0 = time.time()
        E_xc = dft_solver.compute_xc(ngrid, nao, d_dm, d_ao, d_weights, d_vxc, d_ao_grad)
        cp.cuda.Stream.null.synchronize()
        xc_times.append(time.time() - t0)

        J_cpu = d_J.get()
        Vxc_raw = d_vxc.get()
        
        Vxc_cpu = 0.5 * (Vxc_raw + Vxc_raw.T)
        F = Hcore + J_cpu + Vxc_cpu
        F = adiis.update(S, dm, F)
        e, C = eigh(F, S)

        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T

        E_one  = np.sum(dm_new * Hcore)
        E_coul = 0.5 * np.sum(dm_new * J_cpu)
        E_tot  = E_one + E_coul + E_xc + E_nuc
        
        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        
        print(f"{cycle+1:4d} {E_tot:18.8f} {dE:15.6e} {dm_change:15.6e}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            end_time = time.time()
            
            avg_xc = sum(xc_times) / len(xc_times) * 1000 
            
            print("-" * 65)
            print(f"Converged!")
            print(f"Total Energy: {E_tot:.8f} Ha")
            print(f"E_one: {E_one:.8f} Ha")
            print(f"E_coul: {E_coul:.8f} Ha")
            print(f"E_nuc: {E_nuc:.8f} Ha")
            print(f"E_xc: {E_xc:.8f} Ha")
            print(f"Total Time: {end_time - start_time:.4f} s")
            print("-" * 65)
            print(f"Kernel Statistics (Avg per iter):")
            print(f"XC(Exc+Vxc) Time: {avg_xc:.4f} ms")
            print("-" * 65)
            break

        dm = dm_new
        E_old = E_tot

    if not converged:
        print("SCF Unconverged.")

    print("\nRunning PySCF reference calculation...")
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    
    if args.functional == "LDA":
        mf.xc = 'slater,vwn5' 
    else:
        mf.xc = 'PBE,PBE'    

    start = time.time()
    mf.kernel()
    elapsed = time.time() - start

    Etot_ref = mf.e_tot
    print(f'PySCF ({mf.xc}) Energy : {Etot_ref:.8f} Hartree')
    print(f'Difference             : {abs(Etot_ref - E_tot):.2e} Hartree')
    print(f'PySCF Time             : {elapsed:.4f} s')