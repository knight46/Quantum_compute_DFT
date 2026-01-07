

# DFT Functionals with Gaussian Basis Sets

This repository provides CUDA- and CPU-based implementations of Density Functional Theory (DFT) exchange–correlation functionals using **Gaussian basis sets**, including **LDA** and **GGA**.
The CUDA implementations are designed for **GPU acceleration**, while CPU fallbacks are also provided for verification and comparison.

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
g++ -shared -o ./weigths/liblda.so ./src/lda.cpp -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
```

> **Note**
> The CPU version is intended for validation and benchmarking against the CUDA implementation.

---

### Run the Python Script

Execute the Python driver script after successful compilation.

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

## Molecular Data and Grid Files

The repository includes predefined:

* Molecular coordinate files
* Auxiliary data required for DFT evaluations

You may:

* Select existing molecular configurations, or
* Append your own molecular geometries definitions following the existing format.




