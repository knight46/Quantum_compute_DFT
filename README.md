# LDA(Local Density Approximation)

## Compile CUDA Source Files

### Target Architecture 1 (e.g., RTX 4060) - Compute Capability 8.6
`nvcc -shared -o ./weights/lda.so ./src/lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

### Target Architecture 2 (e.g., MX250) - Compute Capability 5.0
`nvcc -shared -o ./weights/mxlda.so ./src.lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50`


## Compile C++ Source Files
`g++ -shared -o ./weigths/liblda.so ./src/lda.cpp -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

## Run Python Script 

Before running the script, update the library path in LDA.py:

 - For Architecture 1: Change the `libname` (LDA.py:Linux) variable to `lda.so`.

 - For Architecture 2: Change the `libname` (LDA.py:Linux) variable to `mxlda.so`.

Execute the script:

`python LDA.py`


# GGA(Generalized Gradient Approximation)

## Compile CUDA Source Files

### Target Architecture 1 (e.g., RTX 4060) - Compute Capability 8.6
`nvcc -shared -o ./weights/gga.so ./src/gga.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

### Target Architecture 2 (e.g., MX250) - Compute Capability 5.0
`nvcc -shared -o ./weights/mxgga.so ./src/gga.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50`

## Run Python Script

Before running the script, update the library path in LDA.py:

 - For Architecture 1: Change the `libname` (GGA.py:Linux) variable to `gga.so`.

 - For Architecture 2: Change the `libname` (GGA.py:Linux) variable to `mxgga.so`.

Execute the script:

`python GGA.py`


**Note:** The files contain various molecular coordinate data and grid information. You may select existing configurations or append custom data as needed