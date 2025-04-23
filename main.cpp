#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include "ndarray.hpp"
#include "wavelet.hpp"
#include "multilevel.hpp"

int main(int argc, char* argv[]) {
    // Usage: wt <input.bin> <num_dims> <dim1> ... <dimN> <wavelet_name> <output.bin>
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <num_dims> <dim1> ... <dimN> <wavelet_name> <output.bin>\n";
        return 1;
    }

    std::string input_path = argv[1];
    int num_dims = std::stoi(argv[2]);
    if (argc < 4 + num_dims) {
        std::cerr << "Error: expected " << num_dims << " dimensions but got fewer arguments.\n";
        return 1;
    }

    // 解析各维度长度
    std::vector<std::size_t> shape(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        shape[i] = std::stoul(argv[3 + i]);
    }
    // 波形名称和输出文件
    std::string wavelet_name = argv[3 + num_dims];
    std::string output_path  = argv[4 + num_dims];

    // 计算总元素数
    std::size_t total = std::accumulate(
        shape.begin(), shape.end(),
        std::size_t(1), std::multiplies<>());

    // 读入二进制数据（double）
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }
    NDArray<double> data(shape);
    in.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
    if (!in) {
        std::cerr << "Failed to read " << total << " elements from " << input_path << "\n";
        return 1;
    }
    in.close();

    try {
        // 初始化小波
        Wavelet w(wavelet_name);

        // 所有轴：0,1,...,num_dims-1
        std::vector<int> axes(num_dims);
        std::iota(axes.begin(), axes.end(), 0);

        // 多尺度分解 & 重构，默认 periodic 模式
        auto coeffs = wavedecn(data, w, "periodic", -1, axes);
        auto rec    = waverecn(coeffs, w, "periodic", axes);

        // 计算 L2 误差
        double l2_error = 0.0;
        for (std::size_t i = 0; i < total; ++i) {
            double diff = data.data()[i] - rec.data()[i];
            l2_error += diff * diff;
        }
        l2_error = std::sqrt(l2_error);
        std::cout << "Reconstruction L2 error: " << l2_error << std::endl;

        // 写出重构结果
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(rec.data()), total * sizeof(double));
        out.close();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}