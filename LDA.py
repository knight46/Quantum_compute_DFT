import numpy as np
import cupy as cp
from scipy.linalg import eigh
import sys, ctypes, os, time
from pyscf import gto, dft, ao2mo
from pyscf.scf import diis
from grid import build
import argparse


libname = {'linux':'./weights/lda.so', 'darwin':'liblda.so', 'win32':'lda.dll'}[sys.platform]
lib = ctypes.CDLL(os.path.abspath(libname))




lib.get_rho_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

lib.build_vxc_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ctypes.c_void_p, ctypes.c_void_p
]

lib.compute_exc_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
lib.compute_exc_gpu.restype = ctypes.c_double

lib.build_coulomb_gpu.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

def load_xyz_as_pyscf_atom(xyz_path):
    with open(xyz_path, "r") as f: lines = f.readlines()
    return "".join(lines[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xyzfile", type=str)
    args = parser.parse_args()


    atom_file = args.xyzfile if args.xyzfile.lower().endswith(".xyz") else args.xyzfile + ".xyz"
    grid_path = f"./grid_txt/{atom_file.replace('.xyz', '')}_grid.txt"
    atom_path = f"./atom_txt/{atom_file}"

    if not os.path.exists(atom_path):
        print(f"Error: {atom_path} not found.")
        exit(1)


    print("Building CPU data...")
    atom_structure = load_xyz_as_pyscf_atom(atom_path)

    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_path)
    
    ngrid, nao = ao_values.shape
    print(f"System Info: NAO={nao}, Grid={ngrid}, Occupied={nocc}")


    if eri.size != nao**4:
        print(f"Detected packed ERI (size {eri.size}), restoring to full tensor ({nao**4})...")
        eri = ao2mo.restore(1, eri, nao)

    print("Moving data to GPU...")
    t_start_gpu = time.time()
    
    d_ao      = cp.asarray(ao_values, dtype=np.float64, order='C')
    d_weights = cp.asarray(grids.weights, dtype=np.float64, order='C')
    d_eri     = cp.asarray(eri.reshape(nao*nao, nao*nao), dtype=np.float64, order='C')
    
    d_dm      = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_J       = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_vxc     = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_rho     = cp.zeros(ngrid, dtype=np.float64)
    
    d_B_work  = cp.zeros((ngrid, nao), dtype=np.float64, order='C')
    d_exc_work= cp.zeros(ngrid, dtype=np.float64)

    cp.cuda.Stream.null.synchronize()
    print(f"GPU Init Time: {time.time() - t_start_gpu:.4f}s")

    e, C = eigh(Hcore, S)
    dm = 2 * C[:, :nocc] @ C[:, :nocc].T
    
    adiis = diis.CDIIS()

    print("\nSCF started! (Hybrid: GPU Integrals + CPU Eigensolver)")
    print("-" * 65)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12}")
    print("-" * 65)

    E_old = 0.0
    converged = False

    p_ao      = d_ao.data.ptr
    p_weights = d_weights.data.ptr
    p_eri     = d_eri.data.ptr
    p_dm      = d_dm.data.ptr
    p_J       = d_J.data.ptr
    p_vxc     = d_vxc.data.ptr
    p_rho     = d_rho.data.ptr
    p_B       = d_B_work.data.ptr
    p_exc_w   = d_exc_work.data.ptr
    
    start_time = time.time()

    vxc_times = []
    exc_times = []

    for cycle in range(200):

        d_dm.set(dm)

        lib.build_coulomb_gpu(nao, p_eri, p_dm, p_J)

        lib.get_rho_gpu(nao, ngrid, p_dm, p_ao, p_rho)
        
        cp.cuda.Stream.null.synchronize()
        t_vxc_start = time.time()
        
        lib.build_vxc_gpu(nao, ngrid, p_ao, p_weights, p_rho, p_B, p_vxc)
        
        cp.cuda.Stream.null.synchronize()
        t_vxc_end = time.time()
        vxc_times.append(t_vxc_end - t_vxc_start)
        
        t_exc_start = time.time()
        
        E_xc = lib.compute_exc_gpu(ngrid, nao, p_weights, p_rho, p_exc_w)
        
        t_exc_end = time.time()
        exc_times.append(t_exc_end - t_exc_start)

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
            
            avg_vxc = sum(vxc_times) / len(vxc_times) * 1000 
            avg_exc = sum(exc_times) / len(exc_times) * 1000 
            
            print("-" * 65)
            print(f"Converged!")
            print(f"Total Energy: {E_tot:.8f} Ha")
            print(f"Total Time: {end_time - start_time:.4f} s")
            print("-" * 65)
            print(f"Performance Statistics (Average per iteration):")
            print(f"Vxc Time: {avg_vxc:.4f} ms")
            print(f"Exc Time : {avg_exc:.4f} ms")
            print("-" * 65)
            print("")
            break

        dm = dm_new
        E_old = E_tot

    if not converged:
        print("SCF Unconverged.")

    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()

    print('PySCF (LDA/SVWN) reference:')
    start = time.time()
    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn' 
    mf.kernel()
    dm_ref = mf.make_rdm1()
    elapsed = time.time() - start

    Etot  = mf.energy_tot()
    h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    E1 = np.einsum('ij,ji->', h1, dm_ref)
    vh = mf.get_j(mol, dm_ref)
    Ecoul = 0.5 * np.einsum('ij,ji->', vh, dm_ref)
    Exc   = mf.energy_elec()[0] - (E1 + Ecoul)
    
    print(f' E_one  : {E1:.6f} Hartree')
    print(f' E_coul : {Ecoul:.6f} Hartree')
    print(f' E_exc  : {Exc:.6f} Hartree')
    print(f' E_tot  : {Etot:.8f} Hartree')
    print(f' time   : {elapsed:.6f} s')