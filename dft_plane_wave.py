#!/usr/bin/env python3
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

Bohr = 1.0
# ---------- 关键：盒子缩小到 6 Å ----------
L   = 6.0 * Bohr        # 原来是 10 Bohr
ke  = 30.0
ng  = 64
Z   = 1.0
rc  = 0.2 * Bohr
pos = np.array([[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.74]]) * Bohr   # 0.74 Å 键长

dr = L / ng
idx = np.arange(ng) - ng//2
x, y, z = np.meshgrid(idx*dr, idx*dr, idx*dr, indexing='ij')
r = np.stack((x, y, z), axis=-1)

k = 2*np.pi * fftfreq(ng, d=dr)
kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
G2 = kx**2 + ky**2 + kz**2
G2[0,0,0] = 1.0
T_G = 0.5 * G2
T_G[G2 > 2*ke] = 0

def vloc_G(G):
    return -Z * 4*np.pi / G**2 * np.exp(-0.5 * (G*rc)**2)

Vloc_G = vloc_G(np.sqrt(G2))

rho_r = np.ones_like(Vloc_G) * 0.01
rho_r *= 2.0 / rho_r.sum() / dr**3

for it in range(50):
    rho_G = fftn(rho_r) / (L**3)

    vh_G = 4*np.pi * rho_G / G2
    vh_G[0,0,0] = 0
    vh_r = np.real(ifftn(vh_G) * L**3)

    rho_pos = np.abs(rho_r)
    vxc_r = - (3/np.pi)**(1/3) * rho_pos**(1/3)
    exc_r = 0.75 * vxc_r

    vtot_G = Vloc_G + vh_G + fftn(vxc_r) / (L**3)
    vtot_r = np.real(ifftn(vtot_G) * L**3)

    H_G = T_G + vtot_G
    idx_min = np.unravel_index(np.argmin(H_G), H_G.shape)
    e0 = H_G[idx_min]

    psi_G = np.zeros_like(H_G, dtype=complex)
    psi_G[idx_min] = 1.0
    psi_r = ifftn(psi_G) * L**3
    rho_r_new = 2 * np.abs(psi_r)**2

    E_kin = 0.5 * np.sum(np.abs(psi_G)**2 * T_G) * dr**3
    E_ha  = 0.5 * np.sum(np.conj(rho_G) * vh_G) * L**3
    E_xc  = np.sum(rho_pos * exc_r) * dr**3
    E_tot = E_kin + np.sum(rho_r * np.real(ifftn(Vloc_G) * L**3)) * dr**3 \
            + E_ha + E_xc

    print(f'Iter {it+1:2d}: E_tot = {E_tot:.6f} Hartree')

    if it > 1 and abs(E_tot - E_old) < 1e-5:
        break
    E_old = E_tot
    rho_r = 0.5 * rho_r + 0.5 * rho_r_new

print('-' * 60)
print(f'Converged PW-LDA energy = {E_tot:.6f} Hartree')