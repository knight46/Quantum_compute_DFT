from pyscf import gto, dft

mol = gto.Mole()

mol.atom = 'H 0.0 0.0 0.0; H 0.0 0.0 0.74'  

# mol.atom = 'O 0.0 0.0 0.0; H 0.0 0.0 0.96; H 0.0 0.93 0.0'
mol.basis = '6-31G'
mol.build()

mf = dft.RKS(mol)
mf.xc = 'LDA,VWN'  
energy = mf.kernel()

print(f'氢气分子的能量: {energy:.6f} Hartree')