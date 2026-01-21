import numpy as np
import cupy as cp
from scipy.linalg import eigh
import sys, ctypes, os, time
from pyscf import gto, dft
from pyscf.scf import diis
import argparse

try:
    from grid import build, get_ao_grad
except ImportError:
    print("Error: grid.py not found or missing required functions.")
    sys.exit(1)

class DFTSolverWrapper:
    TYPE_LDA = 0
    TYPE_GGA = 1
    TYPE_B3LYP = 2

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
            ctypes.c_uint64, 
            ctypes.c_uint64, 
            ctypes.c_uint64, 
            ctypes.c_uint64, 
            ctypes.c_uint64  
        ]
        self.lib.DFT_ComputeXC.restype = ctypes.c_double

        self.lib.DFT_ComputeCoulomb.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64, 
            ctypes.c_uint64, 
            ctypes.c_uint64  
        ]

        if self.functional_type == 'LDA':
            c_type = self.TYPE_LDA
        elif self.functional_type == 'GGA':
            c_type = self.TYPE_GGA
        elif self.functional_type == 'B3LYP':
            c_type = self.TYPE_B3LYP
        else:
            raise ValueError(f"Unsupported functional type: {self.functional_type}")

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

def load_xyz_as_string(xyz_path):
    with open(xyz_path, "r") as f: lines = f.readlines()
    return "".join(lines[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DFT (LDA/GGA/B3LYP) using CUDA backend.")
    parser.add_argument("functional", type=str, choices=["LDA", "GGA", "B3LYP"], help="Functional type")
    parser.add_argument("xyzfile", type=str, help="Molecule name (e.g., H2O)")
    args = parser.parse_args()

    lib_path = "./weights/dft.so"
    atom_file = args.xyzfile if args.xyzfile.lower().endswith(".xyz") else args.xyzfile + ".xyz"
    grid_path = f"./grid_txt/{atom_file.replace('.xyz', '')}_grid.txt"
    atom_path = f"./atom_txt/{atom_file}"

    if not os.path.exists(atom_path):
        print(f"Error: {atom_path} not found.")
        sys.exit(1)

    print(f"=== DFT Solver: {args.functional} | Molecule: {atom_file} ===")

    print("Building CPU data...")
    atom_structure_str = load_xyz_as_string(atom_path)
    
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_path, grid_path)
    
    ngrid, nao = ao_values.shape
    print(f"System Info: NAO={nao}, Grid={ngrid}, Occupied={nocc}")

    ao_grad_cpu = None
    if args.functional in ["GGA", "B3LYP"]:
        print(f"Calculating AO Gradients ({args.functional} mode)...")
        
        try:
            ao_grad_raw = get_ao_grad(atom_structure_str, grids)
        except Exception as e:
            print(f"Error computing gradients: {e}")
            sys.exit(1)

        if ao_grad_raw.shape == (ngrid, nao, 3):
            ao_grad_cpu = np.transpose(ao_grad_raw, (2, 0, 1))
        elif ao_grad_raw.shape == (3, ngrid, nao):
            ao_grad_cpu = ao_grad_raw
        else:
            print(f"Warning: Unexpected ao_grad shape {ao_grad_raw.shape}. Assuming (3, ngrid, nao).")
            ao_grad_cpu = ao_grad_raw
    else:
        print("Skipping AO Gradients (LDA mode).")

    print("Moving data to GPU...")
    t_start_gpu = time.time()

    try:
        dft_solver = DFTSolverWrapper(lib_path, args.functional)
    except Exception as e:
        print(e)
        sys.exit(1)

    d_ao = cp.asarray(ao_values, dtype=cp.float64, order='C')
    
    if hasattr(grids, 'weights'):
        weights_cpu = grids.weights
    elif isinstance(grids, np.ndarray) and grids.shape[1] == 4:
         weights_cpu = grids[:, 3]
    else:
        weights_cpu = np.ones(ngrid)

    d_weights = cp.asarray(weights_cpu, dtype=cp.float64, order='C')
    
    d_eri = cp.asarray(eri.reshape(nao*nao, nao*nao), dtype=cp.float64, order='C')
    
    d_eri_4d = d_eri.reshape(nao, nao, nao, nao)
    
    d_ao_grad = None
    if ao_grad_cpu is not None:
        d_ao_grad = cp.asarray(ao_grad_cpu, dtype=cp.float64, order='C')

    d_dm  = cp.zeros((nao, nao), dtype=cp.float64, order='C')
    d_J   = cp.zeros((nao, nao), dtype=cp.float64, order='C')
    d_vxc = cp.zeros((nao, nao), dtype=cp.float64, order='C')

    cp.cuda.Stream.null.synchronize()
    print(f"GPU Init Time: {time.time() - t_start_gpu:.4f}s")

    e, C = eigh(Hcore, S)
    dm = 2.0 * np.dot(C[:, :nocc], C[:, :nocc].T)
    
    adiis = diis.CDIIS()

    print("\nSCF started!")
    print("-" * 80)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12} {'HF_Ex':>12}")
    print("-" * 80)

    E_old = 0.0
    converged = False
    start_time = time.time()
    
    xc_times = [] 
    
    c_hf = 0.2 if args.functional == "B3LYP" else 0.0

    for cycle in range(200):
        d_dm.set(dm)
        cp.cuda.Stream.null.synchronize()

        dft_solver.compute_coulomb(nao, d_eri, d_dm, d_J)
        
        t0 = time.time()
        E_xc_dft = dft_solver.compute_xc(ngrid, nao, d_dm, d_ao, d_weights, d_vxc, d_ao_grad)
        cp.cuda.Stream.null.synchronize()
        xc_times.append(time.time() - t0)

        J_cpu = d_J.get()
        Vxc_raw = d_vxc.get()
        Vxc_cpu = 0.5 * (Vxc_raw + Vxc_raw.T)
        
        E_ex_hf = 0.0
        K_cpu = 0.0
        
        if args.functional == "B3LYP":
            d_K = cp.einsum('ijkl,jl->ik', d_eri_4d, d_dm)
            K_cpu = d_K.get()
            
            F = Hcore + J_cpu + Vxc_cpu - (c_hf * 0.5 * K_cpu)
        else:
            F = Hcore + J_cpu + Vxc_cpu

        F = adiis.update(S, dm, F)
        
        e, C = eigh(F, S)
        dm_new = 2.0 * np.dot(C[:, :nocc], C[:, :nocc].T)

        E_one  = np.sum(dm_new * Hcore)
        E_coul = 0.5 * np.sum(dm_new * J_cpu)
        
        if args.functional == "B3LYP":
            E_ex_hf = -0.25 * c_hf * np.sum(dm_new * K_cpu)
            
        E_tot  = E_one + E_coul + E_xc_dft + E_ex_hf + E_nuc
        
        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        
        print(f"{cycle+1:4d} {E_tot:18.8f} {dE:15.6e} {dm_change:15.6e} {E_ex_hf:12.6f}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            end_time = time.time()
            
            avg_xc = sum(xc_times) / len(xc_times) * 1000 
            
            print("-" * 80)
            print(f"Converged!")
            print(f"Total Energy: {E_tot:.8f} Ha")
            print(f"E_one       : {E_one:.8f} Ha")
            print(f"E_coul      : {E_coul:.8f} Ha")
            print(f"E_nuc       : {E_nuc:.8f} Ha")
            print(f"E_xc_dft    : {E_xc_dft:.8f} Ha")
            if args.functional == "B3LYP":
                print(f"E_ex_hf     : {E_ex_hf:.8f} Ha")
            print(f"Total Time  : {end_time - start_time:.4f} s")
            print("-" * 80)
            print(f"Kernel Statistics (Avg per iter):")
            print(f"XC(Exc+Vxc) Time: {avg_xc:.4f} ms")
            print("-" * 80)
            break

        dm = dm_new
        E_old = E_tot

    if not converged:
        print("SCF Unconverged.")


    print("\nRunning PySCF reference calculation...")
    mol = gto.Mole()
    mol.atom = atom_structure_str
    
    mol.basis = 'sto-3g'  
    
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    
    if args.functional == "LDA":
        mf.xc = 'slater,vwn5' 
    elif args.functional == "GGA":
        mf.xc = 'PBE,PBE'    
    elif args.functional == "B3LYP":
        mf.xc = 'b3lyp'

    start = time.time()
    mf.kernel()
    elapsed = time.time() - start

    Etot_ref = mf.e_tot
    print(f'PySCF ({mf.xc}) Energy : {Etot_ref:.8f} Hartree')
    print(f'Difference             : {abs(Etot_ref - E_tot):.2e} Hartree')
    print(f'PySCF Time             : {elapsed:.4f} s')