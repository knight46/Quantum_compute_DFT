/*
 * gansu_minimal.cpp
 * 一次性把所有 .cu/.cpp 塞进来，内部链接，对外只留 C 接口
 * nvcc -shared -std=c++17 -Xcompiler=-fPIC -I./GANSU-main/include \
 *   -I/usr/local/cuda/include \
 *   gansu_minimal.cpp \
 *   -o gansu_bridge.so -lcudart
 */
#include <vector>
#include <cstring>
#include <cstdio>

// === 只拷必要头文件 ===
namespace gansu {
struct Atom { int atomic_number; struct{double x,y,z;} coordinate; };
struct PrimitiveShell { double exp,coeff; double x,y,z; int shell_type,basis_idx,atom_idx; };
struct ShellTypeInfo { int count,start_index; };
int element_name_to_atomic_number(const char* n);
} // namespace gansu

// === 底层 GPU 函数声明 ===
extern "C" void computeCoreHamiltonianMatrix(
    const std::vector<gansu::ShellTypeInfo>&,
    gansu::Atom*, gansu::PrimitiveShell*, double*, double*,
    double*, double*, int, int, bool);

/* ---------- 对外 C 接口 ---------- */
extern "C" {

int gansu_compute_SH(const char* atom_str,
                     const char* basis_str,
                     double* S,
                     double* H,
                     int*    nbf_out)
{
    try {
        // 1. 解析原子
        std::vector<gansu::Atom> atoms;
        char* dup = strdup(atom_str);
        char* line = strtok(dup, "\n");
        while (line) {
            char elem[16];
            gansu::Atom a;
            if (sscanf(line, "%s %lf %lf %lf", elem, &a.coordinate.x,
                       &a.coordinate.y, &a.coordinate.z) == 4) {
                a.atomic_number = gansu::element_name_to_atomic_number(elem);
                atoms.push_back(a);
            }
            line = strtok(nullptr, "\n");
        }
        free(dup);

        // 2. 构造分子（这里只支持 sto-3g 内置路径，可扩展）
        gansu::Molecular mol(atoms, basis_str);

        const int n = mol.get_num_basis();
        *nbf_out = n;

        // 3. 分配矩阵
        gansu::DeviceHostMatrix<double> Smat(n, n);
        gansu::DeviceHostMatrix<double> Hmat(n, n);

        // 4. boys 网格
        gansu::DeviceHostMemory<double> boys_grid(30720, true);
        extern const double h_boys_grid[30720];
        for (int i = 0; i < 30720; ++i) boys_grid[i] = h_boys_grid[i];
        boys_grid.toDevice();

        // 5. 上传数据 & 计算
        auto& shell_infos = mol.get_shell_type_infos();
        auto& atoms_vec   = mol.get_atoms();
        auto& shells_vec  = mol.get_primitive_shells();
        auto& norm_vec    = mol.get_cgto_normalization_factors();

        gansu::DeviceHostMemory<gansu::Atom>           d_atoms(atoms_vec.size());
        gansu::DeviceHostMemory<gansu::PrimitiveShell> d_shells(shells_vec.size());
        gansu::DeviceHostMemory<double>                d_norms(norm_vec.size());

        std::copy(atoms_vec.begin(),  atoms_vec.end(),  d_atoms.host_ptr());
        std::copy(shells_vec.begin(), shells_vec.end(), d_shells.host_ptr());
        std::copy(norm_vec.begin(),   norm_vec.end(),   d_norms.host_ptr());

        d_atoms.toDevice();
        d_shells.toDevice();
        d_norms.toDevice();

        computeCoreHamiltonianMatrix(
            shell_infos,
            d_atoms.device_ptr(),
            d_shells.device_ptr(),
            boys_grid.device_ptr(),
            d_norms.device_ptr(),
            Smat.device_ptr(),
            Hmat.device_ptr(),
            mol.get_atoms().size(), n, false);

        // 6. 拷回
        Smat.toHost();
        Hmat.toHost();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                int idx = i + j * n;
                S[idx] = Smat(i, j);
                H[idx] = Hmat(i, j);
            }
        return 0;
    } catch (...) {
        return 1;
    }
}

}   // extern "C"