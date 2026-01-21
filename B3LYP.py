import numpy as np
import cupy as cp
from scipy.linalg import eigh
import sys, ctypes, os, time
from pyscf import gto, dft
from grid import build, get_ao_grad
import argparse

libname = {'linux':'./weights/b3lyp.so', 'darwin':'libb3lyp.so', 'win32':'b3lyp.dll'}[sys.platform]
if not os.path.exists(libname):
    print(f"Error: Library {libname} not found. Please compile B3LYP.cu first.")
    exit(1)

lib = ctypes.CDLL(os.path.abspath(libname))

lib.get_rho_sigma_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

lib.build_vxc_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ctypes.c_void_p, ctypes.c_void_p
]

lib.compute_exc_gpu.argtypes = [
    ctypes.c_int, ctypes.c_int, 
    ctypes.c_void_p, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
lib.compute_exc_gpu.restype = ctypes.c_double

lib.build_coulomb_gpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def adaptive_mixing(dm_new, dm_old, cycle, dm_change):
    if cycle < 5: return 0.5 * dm_new + 0.5 * dm_old
    if dm_change > 1e-2: return 0.3 * dm_new + 0.7 * dm_old
    return 0.5 * dm_new + 0.5 * dm_old

def load_xyz_as_pyscf_atom(xyz_path):
    with open(xyz_path, "r") as f: lines = f.readlines()
    return "".join(lines[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xyzfile", type=str, help="Path to XYZ file (e.g., H2O.xyz)")
    args = parser.parse_args()

    atom_file = args.xyzfile if args.xyzfile.lower().endswith(".xyz") else args.xyzfile + ".xyz"
    grid_path = f"./grid_txt/{atom_file.replace('.xyz', '')}_grid.txt"
    atom_path = f"./atom_txt/{atom_file}"

    if not os.path.exists(atom_path):
        atom_path = atom_file 

    print(f"Reading atom: {atom_path}")
    
    atom_structure = load_xyz_as_pyscf_atom(atom_path)
    
    print("Building CPU data...")
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_path)
    ao_grad = get_ao_grad(atom_structure, grids)
    
    ngrid, nao = ao_values.shape
    print(f"System Info: NAO={nao}, Grid={ngrid}, Occupied={nocc}")

    print("Moving data to GPU...")
    t_start_gpu = time.time()
    
    d_ao          = cp.asarray(ao_values, dtype=np.float64, order='C')
    d_ao_grad     = cp.asarray(ao_grad,   dtype=np.float64, order='C')
    d_weights     = cp.asarray(grids.weights, dtype=np.float64, order='C')
    
    eri_np = eri.reshape(nao, nao, nao, nao)
    d_eri_4d = cp.asarray(eri_np, dtype=np.float64, order='C')
    d_eri_flat = cp.asarray(eri.reshape(nao*nao, nao*nao), dtype=np.float64, order='C')

    d_dm       = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_J        = cp.zeros((nao, nao), dtype=np.float64, order='C')
    d_K        = cp.zeros((nao, nao), dtype=np.float64, order='C') 
    d_vxc      = cp.zeros((nao, nao), dtype=np.float64, order='C')
    
    d_rho      = cp.zeros(ngrid, dtype=np.float64)
    d_sigma    = cp.zeros(ngrid, dtype=np.float64)
    d_grad_rho = cp.zeros((ngrid, 3), dtype=np.float64)
    
    d_B_work   = cp.zeros((ngrid, nao), dtype=np.float64, order='C')
    d_exc_work = cp.zeros(ngrid, dtype=np.float64)

    cp.cuda.Stream.null.synchronize()
    print(f"GPU Init Time: {time.time() - t_start_gpu:.4f}s")

    e, C = eigh(Hcore, S)
    dm = 2 * C[:, :nocc] @ C[:, :nocc].T
    
    print("\nSCF started (B3LYP)!")
    print("-" * 85)
    print(f"{'epoch':>4} {'tot energy':>18} {'Δenergy':>15} {'Δdensity':>15} {'Ex_HF':>12}")
    print("-" * 85)

    E_old = 0.0
    converged = False

    p_ao      = d_ao.data.ptr
    p_ao_grad = d_ao_grad.data.ptr
    p_weights = d_weights.data.ptr
    p_eri_flat= d_eri_flat.data.ptr
    
    p_dm      = d_dm.data.ptr
    p_J       = d_J.data.ptr
    p_vxc     = d_vxc.data.ptr
    p_rho     = d_rho.data.ptr
    p_sigma   = d_sigma.data.ptr
    p_grho    = d_grad_rho.data.ptr
    p_B       = d_B_work.data.ptr
    p_exc_w   = d_exc_work.data.ptr
    
    start_time = time.time()
    vxc_times = []
    exc_times = []
    
    c_hf = 0.2

    for cycle in range(100):
        d_dm.set(dm)

        lib.build_coulomb_gpu(nao, p_eri_flat, p_dm, p_J)

        d_K = cp.einsum('ijkl,jl->ik', d_eri_4d, d_dm)

        lib.get_rho_sigma_gpu(nao, ngrid, p_dm, p_ao, p_ao_grad, p_rho, p_sigma, p_grho)
        
        cp.cuda.Stream.null.synchronize()
        t_vxc_start = time.time()
        
        lib.build_vxc_gpu(nao, ngrid, p_ao, p_ao_grad, p_weights, p_rho, p_sigma, p_grho, p_B, p_vxc)
        
        cp.cuda.Stream.null.synchronize()
        t_vxc_end = time.time()
        vxc_times.append(t_vxc_end - t_vxc_start)
        
        t_exc_start = time.time()
        E_xc_dft = lib.compute_exc_gpu(ngrid, nao, p_weights, p_rho, p_sigma, p_exc_w)
        t_exc_end = time.time()
        exc_times.append(t_exc_end - t_exc_start)

        J_cpu = d_J.get()
        K_cpu = d_K.get()
        Vxc_dft_cpu = d_vxc.get()
        Vxc_dft_cpu = 0.5 * (Vxc_dft_cpu + Vxc_dft_cpu.T)

        F = Hcore + J_cpu + Vxc_dft_cpu - (c_hf * 0.5 * K_cpu)
        
        e, C = eigh(F, S)
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T
        
        E_one  = np.sum(dm_new * Hcore)
        E_coul = 0.5 * np.sum(dm_new * J_cpu)
        E_ex_hf = -0.25 * c_hf * np.sum(dm_new * K_cpu)
        E_tot  = E_one + E_coul + E_xc_dft + E_ex_hf + E_nuc
        
        dE = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)
        
        print(f"{cycle+1:4d} {E_tot:18.8f} {dE:15.6e} {dm_change:15.6e} {E_ex_hf:12.6f}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            end_time = time.time()
            
            avg_vxc = sum(vxc_times) / len(vxc_times) * 1000 
            avg_exc = sum(exc_times) / len(exc_times) * 1000 
            
            print("-" * 85)
            print(f"Converged!")
            print(f"Total Energy: {E_tot:.8f} Ha")
            print(f"  E_one     : {E_one:.8f}")
            print(f"  E_coul    : {E_coul:.8f}")
            print(f"  E_ex_hf   : {E_ex_hf:.8f}")
            print(f"  E_xc_dft  : {E_xc_dft:.8f}")
            print(f"Total Time: {end_time - start_time:.4f} s")
            print("-" * 65)
            print(f"Performance Statistics (Average per iteration):")
            print(f"Vxc Time: {avg_vxc:.4f} ms")
            print(f"Exc Time : {avg_exc:.4f} ms")
            print("-" * 65)
            print("")
            break

        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("SCF Unconverged.")


    print("\n" + "="*85)
    print("ANALYSIS & VERIFICATION (Real CUDA Output vs PySCF)")
    print("="*85)
    s_time = time.time()
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.unit = 'Angstrom' 
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.grids.prune = None 
    mf.kernel()
    dm_ref = mf.make_rdm1()
    
    print(f"PySCF Ref Total Energy : {mf.e_tot:.8f} Ha")
    print(f"CUDA  Code Total Energy: {E_tot:.8f} Ha")
    print(f"Difference             : {E_tot - mf.e_tot:.8f} Ha")
    print(f'PySCF_time             : {time.time() - s_time:.4f} s')
    print("-" * 85)

    lib.get_xc_components_gpu.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    c_out_components = (ctypes.c_double * 4)()
    
    p_weights = d_weights.data.ptr
    p_rho     = d_rho.data.ptr
    p_sigma   = d_sigma.data.ptr
    
    lib.get_xc_components_gpu(ngrid, nao, p_weights, p_rho, p_sigma, c_out_components)
    
    raw_lda_cuda = c_out_components[0]
    raw_b88_cuda = c_out_components[1]
    raw_vwn_cuda = c_out_components[2]
    raw_lyp_cuda = c_out_components[3]

    COEFF_LDA = 0.80
    COEFF_B88 = 0.72
    COEFF_VWN = 0.19
    COEFF_LYP = 0.81

    e_lda_cuda_final = COEFF_LDA * raw_lda_cuda
    e_b88_cuda_final = COEFF_B88 * raw_b88_cuda
    e_vwn_cuda_final = COEFF_VWN * raw_vwn_cuda
    e_lyp_cuda_final = COEFF_LYP * raw_lyp_cuda

    rho_gpu_final = d_rho.get()
    grad_gpu_final = d_grad_rho.get().T
    weights_cpu = d_weights.get()
    
    rho_input_lda = rho_gpu_final.reshape(1, -1)
    rho_input_gga = np.vstack([rho_gpu_final, grad_gpu_final])

    def get_pyscf_value(xc_code, is_gga=False):
        inp = rho_input_gga if is_gga else rho_input_lda
        exc, _, _, _ = dft.libxc.eval_xc(xc_code, inp, spin=0, verbose=0)

        return np.dot(exc * rho_gpu_final, weights_cpu)

    ref_raw_lda = get_pyscf_value("LDA_X")

    ref_raw_b88_full = get_pyscf_value("B88", is_gga=True)
    ref_raw_b88_corr = ref_raw_b88_full - ref_raw_lda
    
    ref_raw_vwn = get_pyscf_value("VWN3") 
    ref_raw_lyp = get_pyscf_value("GGA_C_LYP", is_gga=True)

    e_lda_ref = COEFF_LDA * ref_raw_lda
    e_b88_ref = COEFF_B88 * ref_raw_b88_corr
    e_vwn_ref = COEFF_VWN * ref_raw_vwn
    e_lyp_ref = COEFF_LYP * ref_raw_lyp
    rho_input = np.vstack([rho_gpu_final, grad_gpu_final])
    exc, vxc, _, _ = dft.libxc.eval_xc('GGA_C_LYP', rho_input, spin=0, verbose=0)
    dft_exc = 0.81 * np.dot(exc * rho_gpu_final, weights_cpu)

    print(f"{'Component':<20} {'CUDA Output (Ha)':<25} {'PySCF Ref (Ha)':<25} {'Diff (Ha)':<12}")
    print("-" * 85)
    
    print(f"{'LDA Exchange':<20} {e_lda_cuda_final:18.8f} {e_lda_ref:18.8f} {e_lda_cuda_final - e_lda_ref:22.8f}")
    print(f"{'B88 Correction':<20} {e_b88_cuda_final:18.8f} {e_b88_ref:18.8f} {e_b88_cuda_final - e_b88_ref:22.8f}")
    print(f"{'VWN Correlation':<20} {e_vwn_cuda_final:18.8f} {e_vwn_ref:18.8f} {e_vwn_cuda_final - e_vwn_ref:22.8f}")
    print(f"{'LYP Correlation':<20} {e_lyp_cuda_final:18.8f} {e_lyp_ref:18.8f} {e_lyp_cuda_final - e_lyp_ref:22.8f}")
    
    print("-" * 85)
    total_xc_cuda = e_lda_cuda_final + e_b88_cuda_final + e_vwn_cuda_final + e_lyp_cuda_final
    total_xc_ref  = e_lda_ref + e_b88_ref + e_vwn_ref + e_lyp_ref
    
    print(f"{'Total XC Energy':<20} {total_xc_cuda:18.8f} {total_xc_ref:18.8f} {total_xc_cuda - total_xc_ref:22.8f}")
    print(f"{'Real E_xc_dft':<20} {E_xc_dft:25.8f} (From SCF loop)")

    if abs(total_xc_cuda - E_xc_dft) > 1e-6:
        print("\n[Warning] Analysis sum does not match SCF loop sum. Check coefficients or integration.")
    else:
        print("\n[Success] Analysis breakdown matches the total energy used in SCF.")