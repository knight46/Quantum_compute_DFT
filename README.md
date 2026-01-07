
# DFT Functionals with Gaussian Basis Sets (CUDA Accelerated)

This repository provides high-performance **CUDA-based implementations** of Density Functional Theory (DFT) exchange–correlation functionals using **Gaussian basis sets**. It supports both **LDA (VWN)** and **GGA (PBE)** functionals.

The project has been unified into a single solver engine (`dft_solver.cu`) and a unified Python driver (`dft.py`), designed for **GPU acceleration**.

---

## Prerequisites

* **Hardware**: NVIDIA GPU (Compute Capability 5.0+ recommended).
* **Software**:
* CUDA Toolkit (e.g., 11.x or 12.x)
* Python 3.8+
* Python Libraries: `numpy`, `cupy`, `pyscf`, `scipy`



---

## Compilation

The core calculation kernel is written in CUDA C++. You must compile it into a shared library (`.so`) before running the Python script.

### 1. Determine Your GPU Architecture

Choose the correct `-arch` flag for your GPU to ensure correctness and optimal performance:

* **NVIDIA A100**: `sm_80` (Recommended for this repo)
* **RTX 3090 / 4060 / A6000**: `sm_86` or `sm_89`
* **V100**: `sm_70`

### 2. Compile Command

Run the following command in the project root directory.
*(Example below uses `sm_80` for A100 and CUDA 12.8 env)*

```bash
# Ensure output directory exists
mkdir -p ./weights

# Compile the unified solver
nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_80 -I./eigen_lib -I./src -lcublas ./src/dft_solver.cu -o ./weights/dft.so
```

> **Note:**
> * `-I./src` is required to include `dft_solver.h`.
> * `-lcublas` is required for matrix operations.
> 
> 

---

## Usage

Use the unified `dft.py` script to run calculations. The script automatically handles memory allocation and selects the appropriate C++ solver (LDA or GGA) based on your input.

### Syntax

```bash
python dft.py <Functional> <Molecule>
```

* **Functional**: `LDA` or `GGA`
* **Molecule**: The name of the molecule (corresponding to files in `atom_txt/`).

### Examples

**1. Run LDA calculation on Water (H₂O):**

* Uses `LDASolver` (No gradients calculated, faster).
* Compares against PySCF `slater,vwn5`.

```bash
python dft.py LDA H2O
```

**2. Run GGA calculation on Benzene:**

* Uses `GGASolver` (Calculates AO gradients).
* Compares against PySCF `PBE,PBE`.

```bash
python dft.py GGA Benzene
```

---

## Directory Structure

* `src/`: Contains C++ source code.
* `dft_solver.cu`: Main CUDA implementation for both LDA and GGA.
* `dft_solver.h`: Header file defining the `XCSolver` interface.


* `weights/`: Destination for the compiled shared library (`dft.so`).
* `atom_txt/`: Molecular geometry files (`.xyz`).
* `grid_txt/`: Pre-computed grid information files(For other tests).
* `dft.py`: Main Python entry point.

