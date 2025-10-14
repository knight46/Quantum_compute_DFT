from pyscf import gto, dft

# --- 定义分子 ---
mol = gto.Mole()
mol.verbose = 0   # 不打印默认 SCF 输出
mol.unit = 'Angstrom'
# mol.atom = """
# O 0.0 0.0 0.0
# H 0.0 0.0 0.96
# H 0.0 0.93 0.0
# """
mol.atom = 'H 0.0 0.0 0.0; H 0.0 0.0 0.74'  
# mol.atom = """
# C   0.000000  0.000000  0.000000
# H   0.629118  0.629118  0.629118
# H  -0.629118 -0.629118  0.629118
# H  -0.629118  0.629118 -0.629118
# H   0.629118 -0.629118 -0.629118
# """
mol.basis = 'sto-6g'
mol.charge = 0
mol.spin = 0
mol.build()

# --- B3LYP 计算 ---
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

# --- 分解能量 ---
E_one = mf.energy_elec()[0] - (mf.get_veff(mol, mf.make_rdm1()).exc + mf.energy_nuc())
E_coul = mf.energy_elec()[1] - mf.get_veff(mol, mf.make_rdm1()).exc
E_xc = mf.get_veff(mol, mf.make_rdm1()).exc
E_nuc = mf.energy_nuc()
E_tot = mf.e_tot

# --- 打印结果 ---
print(f"E_one : {E_one:.12f} Hartree")
print(f"E_coul: {E_coul:.12f} Hartree")
print(f"E_xc  : {E_xc:.12f} Hartree")
print(f"E_nuc : {E_nuc:.12f} Hartree")
print(f"SCF converged, E_tot = {E_tot:.12f} Hartree")
