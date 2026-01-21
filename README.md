# DFT Functionals with Gaussian Basis Sets

This repository provides **CUDA- and CPU-based implementations of Density Functional Theory (DFT) exchange–correlation functionals** using **Gaussian basis sets**, including **LDA**, **GGA**, and **B3LYP**.

The CUDA implementations are designed for **GPU acceleration**, while CPU fallbacks (where available) are provided for **verification and benchmarking** against the GPU results.

---

## LDA (Local Density Approximation)

### Compile CUDA Source Files

> **Note:**
> GPU architectures differ across NVIDIA devices. Although precompiled shared libraries (`.so`) are provided in the `weights/` directory, **it is strongly recommended to recompile the CUDA source code for your specific GPU architecture** to ensure correctness and optimal performance.

Typical compute capability mappings:

* **RTX 4060, RTX A6000** → `sm_86`
* **RTX A100** → `sm_80`
* **MX250** → `sm_50`

---

### CUDA (GPU) Version

Example: compiling for **sm_86**

```bash
nvcc -O3 --use_fast_math -shared -o ./weights/lda.so ./src/lda.cu -Xcompiler -fPIC -I./eigen_lib -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -lcublas -lcusolver
```

---

### C++ (CPU) Version

```bash
g++ -shared -o ./weights/liblda.so ./src/lda.cpp -I./eigen_lib
```

> **Note:**
> The CPU version is intended for validation and benchmarking against the CUDA implementation.

---

### Run the Python Script

Example: **water molecule (H₂O)**

```bash
python LDA.py H2O
```

---

## GGA (Generalized Gradient Approximation)

### Compile CUDA Source Files

Example: compiling for **sm_86**

```bash
nvcc -O3 --use_fast_math -shared -o ./weights/gga.so ./src/gga.cu -Xcompiler -fPIC -I./eigen_lib -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -lcublas
```

---

### Run the Python Script

Example: **benzene molecule**

```bash
python GGA.py Benzene
```

---

## B3LYP (Hybrid GGA Functional)

### Overview

**B3LYP** is a widely used **hybrid density functional**, combining:

* Local Density Approximation (LDA)
* Generalized Gradient Approximation (GGA)
* A fraction of **exact Hartree–Fock exchange**

This implementation evaluates the **exchange–correlation energy and potential on numerical grids** using Gaussian basis sets, with CUDA acceleration for high-performance GPU execution.

> ⚠️ Note:
> Since B3LYP includes **exact exchange**, its computational cost is significantly higher than pure LDA or GGA functionals.

---

### Compile CUDA Source Files

Example: compiling for **sm_86**

```bash
nvcc -O3 --use_fast_math -shared -o ./weights/b3lyp.so ./src/B3LYP.cu -Xcompiler -fPIC -I./eigen_lib -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -lcublas
```

---

### Run the Python Script

Example: **water molecule (H₂O)**

```bash
python B3LYP.py H2O
```

---

## Molecular Data and Grid Files

The repository includes predefined:

* Molecular coordinate files
* Numerical integration grids
* Auxiliary data required for DFT evaluations

You may:

* Select existing molecular configurations, or
* Append your own molecular geometry definitions following the existing format.

---

## Notes

* GPU and CPU results are intended to be numerically consistent within acceptable floating-point tolerances.
* Recompilation is recommended when switching GPU architectures.
* This codebase is designed primarily for **research, verification, and educational purposes**.

