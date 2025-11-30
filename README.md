# LDA

## 编译 CUDA 文件

### arch架构1 (4060...)
`nvcc -shared -o lda.so lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

### arch架构2 (MX250...)
`nvcc -shared -o mxlda.so lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50`


## 编译 C++ 文件
`g++ -shared -o liblda.so lda.cpp -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

## 运行 Python 脚本  

使用arch架构1需要将文件LDA.py文件中上面libname中的linux改为lda.so

使用arch架构2需要将文件LDA.py文件中上面libname中的linux改为mxlda.so

`python LDA.py`


# GGA
## 编译 CUDA 文件

### arch架构1 (4060...)
`nvcc -shared -o gga.so gga.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86`

### arch架构2 (MX250...)
`nvcc -shared -o mxgga.so gga.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50`

## 运行Python脚本

使用arch架构1需要将文件GGA.py文件中上面libname中的linux改为gga.so

使用arch架构2需要将文件GGA.py文件中上面libname中的linux改为mxgga.so

`python GGA.py`


**PS：** 文件类有各种分子坐标信息以及网格信息,可自行选择或添加
