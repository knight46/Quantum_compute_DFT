# 编译 CUDA 文件

## arch架构1 (4060...)
nvcc -shared -o lda.so lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

## arch架构2 (MX250...)
nvcc -shared -o mxlda.so lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50


# 编译 C++ 文件
g++ -shared -o liblda.so lda.cpp -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

# 运行 Python 脚本  
python LDA.py
