# 编译 CUDA 文件
nvcc -shared -o lda.so lda.cu -Xcompiler -fPIC -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

# 编译 C++ 文件
g++ -shared -o liblda.so lda.cpp -I/usr/include/eigen3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

# 运行 Python 脚本
python LDA.py
