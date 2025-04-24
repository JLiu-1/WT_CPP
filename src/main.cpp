// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "sym13_wavedecn.hpp"  // 你的全部 NDArray, wavedecn_simple, waverecn_simple, save_coeffs_binary

// 从 binary 文件读取带 header 的 NDArray<T>
template<typename U>
NDArray<U> load_ndarray_binary(const std::string& fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file: " + fname);
    // header: uint32 ndim, followed by ndim x uint64 dims
    uint32_t ndim;
    in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint64_t d;
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        shape[i] = static_cast<size_t>(d);
    }
    NDArray<U> arr(shape);
    in.read(reinterpret_cast<char*>(arr.data()), arr.size() * sizeof(U));
    return arr;
}

// 仅把原始数据写入，不写维度等 header
template<typename U>
void save_ndarray_raw(const std::string& fname, const U* data, size_t count) {
    std::ofstream out(fname, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file: " + fname);
    out.write(reinterpret_cast<const char*>(data), count * sizeof(U));
}

template<typename U>
void write_bytes(std::ofstream& out, const U* data, size_t count) {
    out.write(reinterpret_cast<const char*>(data), count * sizeof(U));
}

// 二进制保存 NDArray<T>：
// [uint32 ndim]
// [uint64 dims[0] ... dims[ndim-1]]
// [T data[0] ... data[size-1]]
template<typename T>
void save_ndarray_binary(const std::string& fname, const NDArray<T>& arr) {
    std::ofstream out(fname, std::ios::binary);
    if (!out) throw std::runtime_error("无法打开文件: " + fname);
    uint32_t ndim = static_cast<uint32_t>(arr.ndim());
    write_bytes(out, &ndim, 1);
    // dims
    for (size_t d : arr.shape()) {
        uint64_t dd = static_cast<uint64_t>(d);
        write_bytes(out, &dd, 1);
    }
    // raw data
    write_bytes(out, arr.data(), arr.size());
}

// 二进制保存 WaveCoeffs<T>
// 格式：
//   uint32_t num_levels
//   for each level:
//     uint32_t num_entries
//     for each entry:
//       uint32_t key_len
//       char     key[key_len]
//       uint32_t ndim
//       uint64_t dims[ndim]
//       T        data[size]
//   // 最后写最高层 cA：
//   uint32_t marker = 0xFFFFFFFF
//   uint32_t ndim_cA
//   uint64_t dims_cA[...]
//   T        data_cA[...]
template<typename T>
void save_coeffs_binary(const std::string& fname, const WaveCoeffs<T>& wc) {
    std::ofstream out(fname, std::ios::binary);
    if (!out) throw std::runtime_error("无法打开文件: " + fname);

    // levels
    uint32_t num_levels = static_cast<uint32_t>(wc.details.size());
    write_bytes(out, &num_levels, 1);

    for (auto const& level_map : wc.details) {
        uint32_t num_entries = static_cast<uint32_t>(level_map.size());
        write_bytes(out, &num_entries, 1);
        for (auto const& kv : level_map) {
            const std::string& key = kv.first;
            auto const& arr = kv.second;
            // key
            uint32_t key_len = static_cast<uint32_t>(key.size());
            write_bytes(out, &key_len, 1);
            out.write(key.data(), key_len);
            // dims
            uint32_t ndim = static_cast<uint32_t>(arr.ndim());
            write_bytes(out, &ndim, 1);
            for (size_t d : arr.shape()) {
                uint64_t dd = static_cast<uint64_t>(d);
                write_bytes(out, &dd, 1);
            }
            // data
            write_bytes(out, arr.data(), arr.size());
        }
    }

    // cA 层
    uint32_t marker = 0xFFFFFFFF;
    write_bytes(out, &marker, 1);

    auto const& a = wc.cA;
    uint32_t ndim = static_cast<uint32_t>(a.ndim());
    write_bytes(out, &ndim, 1);
    for (size_t d : a.shape()) {
        uint64_t dd = static_cast<uint64_t>(d);
        write_bytes(out, &dd, 1);
    }
    write_bytes(out, a.data(), a.size());
}



int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <dtype: float|double>"
                  << " <dim1,dim2,...> <mode: symmetric|periodic>"
                  << " <fwd_out.bin> <inv_out.bin>\n";
        return 1;
    }

    std::string in_file   = argv[1];
    std::string dtype     = argv[2];
    std::string dim_str   = argv[3];
    std::string mode      = argv[4];
    std::string fwd_file  = argv[5];
    std::string inv_file  = argv[6];

    // 解析维度字符串 "64,64,32" → vector<size_t>{64,64,32}
    std::cout<<dim_str<<std::endl;
    std::vector<size_t> shape;
    {
        std::stringstream ss(dim_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            shape.push_back(static_cast<size_t>(std::stoul(token)));
        }
    }

    try {
        if (dtype == "float") {
            // 读入原始 float32 数据
            auto data_f = load_ndarray_binary<float>(in_file);
            std::cout<<"p1"<<std::endl;
            // 验证维度是否一致
            if (data_f.shape() != shape)
                throw std::runtime_error("Input shape mismatch");
            std::cout<<"p2"<<std::endl;
            // 正向 DWT 分解
            auto coeffs = wavedecn_simple<float>(data_f, mode);
            std::cout<<"p3"<<std::endl;
            // 保存系数（包含 header）
            save_coeffs_binary(fwd_file, coeffs);
            std::cout<<"p4"<<std::endl;
            // 逆向重构
            auto rec = waverecn_simple<float>(coeffs, mode);
            std::cout<<"p5"<<std::endl;
            // 打印重构后的维度
            std::cout << "Reconstructed shape:";
            for (auto d : rec.shape()) std::cout << ' ' << d;
            std::cout << "\n";

            // 只写 raw 数据
            save_ndarray_raw(inv_file, rec.data(), rec.size());
        }
        else if (dtype == "double") {
            // 读入原始 float64 数据
            auto data_d = load_ndarray_binary<double>(in_file);
            if (data_d.shape() != shape)
                throw std::runtime_error("Input shape mismatch");

            auto coeffs = wavedecn_simple<double>(data_d, mode);
            save_coeffs_binary(fwd_file, coeffs);
            auto rec = waverecn_simple<double>(coeffs, mode);

            std::cout << "Reconstructed shape:";
            for (auto d : rec.shape()) std::cout << ' ' << d;
            std::cout << "\n";

            save_ndarray_raw(inv_file, rec.data(), rec.size());
        }
        else {
            throw std::invalid_argument("Unsupported dtype: " + dtype);
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}