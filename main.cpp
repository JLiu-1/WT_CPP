#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <stdexcept>

#include "ndarray.hpp"
#include "wavelet.hpp"
#include "multilevel.hpp"

int main(int argc, char* argv[]) {
    // Usage: wt <input.bin> <float|double> <num_dims> <dim1> ... <dimN> <wavelet_name> <output.bin>
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <float|double> <num_dims> <dim1> ... <dimN> <wavelet_name> <output.bin>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string type_str   = argv[2];
    int num_dims           = std::stoi(argv[3]);
    if (argc < 5 + num_dims) {
        std::cerr << "Error: expected " << num_dims << " dimensions but got fewer arguments.\n";
        return 1;
    }

    std::vector<std::size_t> shape(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        shape[i] = std::stoul(argv[4 + i]);
    }
    std::string wavelet_name = argv[4 + num_dims];
    std::string output_path  = argv[5 + num_dims];

    std::size_t total = std::accumulate(
        shape.begin(), shape.end(), std::size_t(1), std::multiplies<>());

    try {
        if (type_str == "float") {
            NDArray<float> data(shape);
            std::ifstream in(input_path, std::ios::binary);
            if (!in) throw std::runtime_error("Failed to open input file: " + input_path);
            in.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
            if (!in) throw std::runtime_error("Failed to read data from " + input_path);
            in.close();

            Wavelet w(wavelet_name);
            std::vector<int> axes(num_dims);
            std::iota(axes.begin(), axes.end(), 0);

            auto coeffs = wavedecn(data, w, "periodic", -1, axes);
            auto rec    = waverecn(coeffs, w, "periodic", axes);

            double l2_error = 0.0;
            for (std::size_t i = 0; i < total; ++i) {
                double diff = static_cast<double>(data.data()[i] - rec.data()[i]);
                l2_error += diff * diff;
            }
            l2_error = std::sqrt(l2_error);
            std::cout << "Reconstruction L2 error (float): " << l2_error << std::endl;

            std::ofstream out(output_path, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open output file: " + output_path);
            out.write(reinterpret_cast<const char*>(rec.data()), total * sizeof(float));
            out.close();

        } else if (type_str == "double") {
            NDArray<double> data(shape);
            std::ifstream in(input_path, std::ios::binary);
            if (!in) throw std::runtime_error("Failed to open input file: " + input_path);
            in.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
            if (!in) throw std::runtime_error("Failed to read data from " + input_path);
            in.close();

            Wavelet w(wavelet_name);
            std::vector<int> axes(num_dims);
            std::iota(axes.begin(), axes.end(), 0);

            auto coeffs = wavedecn(data, w, "periodic", -1, axes);
            auto rec    = waverecn(coeffs, w, "periodic", axes);

            double l2_error = 0.0;
            for (std::size_t i = 0; i < total; ++i) {
                double diff = data.data()[i] - rec.data()[i];
                l2_error += diff * diff;
            }
            l2_error = std::sqrt(l2_error);
            std::cout << "Reconstruction L2 error (double): " << l2_error << std::endl;

            std::ofstream out(output_path, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open output file: " + output_path);
            out.write(reinterpret_cast<const char*>(rec.data()), total * sizeof(double));
            out.close();

        } else {
            std::cerr << "Unsupported data type: " << type_str << std::endl;
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}