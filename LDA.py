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
    # atom_structure ="""
    #     C   0.000    0.000    0.000
    #     O   0.000    0.000    1.230
    #     O   1.230    0.000   -0.650
    #     H   1.750    0.890   -0.200
    #     C  -1.430    0.000    0.000
    #     C  -2.150    1.235    0.000
    #     C  -3.580    1.235    0.000
    #     C  -4.300    2.470    0.000
    #     C  -5.730    2.470    0.000
    #     C  -6.450    3.705    0.000
    #     C  -7.880    3.705    0.000
    #     C  -8.600    4.940    0.000
    #     C  -10.030   4.940    0.000
    #     C  -10.750   6.175    0.000
    #     C  -12.180   6.175    0.000
    #     C  -12.900   7.410    0.000
    #     C  -14.330   7.410    0.000
    #     C  -15.050   8.645    0.000
    #     C  -16.480   8.645    0.000
    #     C  -17.200   9.880    0.000
    #     C  -18.630   9.880    0.000
    #     C  -19.350   11.115   0.000
    #     C  -20.780   11.115   0.000
    #     C  -21.500   12.350   0.000
    #     C  -22.930   12.350   0.000
    #     H  -23.450   13.300   0.000
    # """
    # atom_structure = 'O 0.0 0.0 0.0; H 0.0 0.0 0.96; H 0.0 0.93 0.0'

    # atom_structure ="""
    #     S          4.69233        0.02478       -3.55541
    #     S          6.71441        0.50263       -2.99069
    #     O         -0.13939        2.76881       -0.92057
    #     O          2.09514       -1.44577       -2.83015
    #     O         -3.88192       -0.93091       -1.53062
    #     O          1.96357        1.88375        1.57555
    #     O          7.13930        1.15792        0.88067
    #     O          2.27726        5.17877        0.03236
    #     O          6.84799        4.33262       -3.39628
    #     O         11.31591        3.36805       -1.17437
    #     O         -5.43340        3.17967        1.67198
    #     O          8.80127        3.63918       -5.60597
    #     O          9.35553        5.03529        2.30173
    #     O         16.09379        6.45004       -2.63364
    #     N          0.35198       -0.04724       -2.86291
    #     N         -1.30758        0.99538       -0.28831
    #     N          2.12933        0.87675       -0.40377
    #     N         -4.35218        0.39960        0.17303
    #     N          5.03143        1.82231        0.63049
    #     N          6.75817        3.50249       -1.34541
    #     N         -3.08453       -3.22833        2.19796
    #     N          9.57449        4.70466       -1.41748
    #     N          9.38047        2.43730       -3.79699
    #     N          1.78061        4.45544       -2.01444
    #     N          7.92305        1.29269       -6.82560
    #     N         -3.93248        3.36234        0.03659
    #     N         -5.00705       -3.24591        0.76970
    #     N         -3.75923       -5.27600        1.31269
    #     N          9.08903        2.99758        3.16124
    #     C         -0.32285        1.23544       -2.67942
    #     C         -1.61548        1.11051       -3.49794
    #     C         -1.18387        0.16874       -4.60342
    #     C         -0.38173       -0.86406       -3.81900
    #     C         -0.59282        1.68135       -1.23971
    #     C          1.64323       -0.35648       -2.49844
    #     C          2.58107        0.63196       -1.78092
    #     C         -1.91079       -0.32804       -0.41410
    #     C          4.04523        0.12879       -1.84247
    #     C         -1.32508       -1.35128        0.59893
    #     C         -3.44036       -0.29723       -0.58106
    #     C         -1.66370       -1.23394        2.09628
    #     C          2.59091        1.78615        0.52981
    #     C          3.84375        2.67760        0.41836
    #     C         -3.03715       -1.80427        2.48446
    #     C          3.79973        3.59398       -0.84058
    #     C          7.06412        3.34555        0.07833
    #     C          6.38563        2.06990        0.56682
    #     C         -4.12163        1.24722        1.32866
    #     C          6.70976        4.55622        0.99315
    #     C          7.43963        0.31195       -4.65291
    #     C          2.55998        4.45071       -0.90785
    #     C          8.38173        5.37148       -1.94257
    #     C          7.58055        1.63758       -5.44003
    #     C          7.28732        4.37119       -2.25501
    #     C         10.55537        3.20932       -3.36789
    #     C          7.10479        4.33039        2.46545
    #     C          8.67236        6.34826       -3.10151
    #     C         -4.53809        2.66050        1.02235
    #     C         10.45847        3.78645       -1.94234
    #     C          8.63174        2.63701       -4.92951
    #     C         11.20796        4.16512       -4.41371
    #     C         -3.92540       -3.87529        1.45259
    #     C          7.55761        7.32001       -3.46928
    #     C          8.59189        4.14203        2.63681
    #     C         12.51586        4.76141       -3.93358
    #     C          6.31136        7.38780       -2.80375
    #     C          7.79248        8.20006       -4.54140
    #     C         12.64567        6.15520       -3.79172
    #     C         13.61864        3.94111       -3.63434
    #     C          5.33970        8.30760       -3.20945
    #     C          6.81669        9.11741       -4.94164
    #     C          5.59113        9.17077       -4.27658
    #     C         13.84884        6.71482       -3.35347
    #     C         14.82143        4.50408       -3.19658
    #     C         14.93979        5.89386       -3.05399
    #     H          0.30775        2.00000       -3.18853
    #     H         -2.39950        0.58330       -2.92852
    #     H         -2.03872        2.08081       -3.83457
    #     H         -2.00989       -0.25366       -5.21385
    #     H         -0.48641        0.72267       -5.27320
    #     H          0.26777       -1.44100       -4.51319
    #     H         -1.05563       -1.56442       -3.28508
    #     H          2.51724        1.56636       -2.36314
    #     H         -1.37491        1.46418        0.62087
    #     H         -1.54823       -0.76143       -1.35269
    #     H          4.66941        0.87898       -1.38762
    #     H          4.16771       -0.83940       -1.31186
    #     H          1.31224        0.32698       -0.10471
    #     H         -1.56527       -2.38458        0.26585
    #     H         -0.22213       -1.25600        0.51391
    #     H         -0.92073       -1.84712        2.65335
    #     H         -1.53115       -0.18908        2.44090
    #     H          3.83181        3.37484        1.28542
    #     H         -3.16661       -1.67310        3.57942
    #     H         -3.84137       -1.24888        2.00626
    #     H          4.60910        4.33024       -0.81204
    #     H          3.91406        3.02827       -1.77623
    #     H         -5.34081        0.33004       -0.11800
    #     H          4.81015        0.85384        0.91963
    #     H          8.16055        3.18092        0.15106
    #     H         -4.72992        0.86297        2.17615
    #     H         -3.08853        1.23897        1.67415
    #     H          7.21740        5.47396        0.66675
    #     H          5.63032        4.77409        0.98512
    #     H          6.78424       -0.38805       -5.21269
    #     H          8.41515       -0.21407       -4.57930
    #     H          8.02582        6.04083       -1.14338
    #     H          6.08652        2.83683       -1.74509
    #     H          6.58821        2.14244       -5.44644
    #     H         11.34603        2.42803       -3.29934
    #     H          6.79660        5.21647        3.06139
    #     H          6.55105        3.45452        2.86647
    #     H          9.56412        6.95589       -2.83460
    #     H          8.85959        5.76767       -4.00565
    #     H          9.76147        4.95504       -0.43235
    #     H         11.41424        3.59623       -5.34692
    #     H         10.55859        4.98870       -4.70307
    #     H          9.18638        1.61079       -3.21812
    #     H          2.01023        3.86949       -2.82954
    #     H          0.94520        5.05745       -2.06175
    #     H          7.18604        0.66884       -7.22843
    #     H          7.92781        2.15443       -7.41838
    #     H          6.06731        6.74252       -1.97320
    #     H          8.73696        8.17331       -5.07156
    #     H         -3.17962        2.94223       -0.52542
    #     H         -4.22062        4.33035       -0.16919
    #     H         11.81594        6.81297       -4.01843
    #     H         13.55400        2.86641       -3.74258
    #     H         -5.63727       -3.81698        0.18854
    #     H         -5.21547       -2.24347        0.82534
    #     H         -2.98752       -5.76213        1.79245
    #     H         -4.40672       -5.82475        0.72878
    #     H          8.46352        2.23179        3.45090
    #     H         10.10592        2.87463        3.27676
    #     H          4.38844        8.35029       -2.69430
    #     H          7.01130        9.78716       -5.76934
    #     H          4.83601        9.88107       -4.58791
    #     H         13.93268        7.78926       -3.24778
    #     H         15.66101        3.85883       -2.97042
    #     H         16.87283        5.91815       -2.41764
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

    for cycle in range(200):
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

