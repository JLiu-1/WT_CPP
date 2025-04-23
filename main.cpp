#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include "ndarray.hpp"
#include "wavelet.hpp"
#include "multilevel.hpp"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <num_elements> <wavelet_name> <output.bin>\n";
        return 1;
    }

    std::string input_path = argv[1];
    size_t N = std::stoul(argv[2]);
    std::string wavelet_name = argv[3];
    std::string output_path = argv[4];

    // Read raw binary input (double precision)
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }
    NDArray<double> data({N});
    in.read(reinterpret_cast<char*>(data.data()), N * sizeof(double));
    if (!in) {
        std::cerr << "Failed to read " << N << " elements from " << input_path << "\n";
        return 1;
    }
    in.close();

    try {
        // Initialize wavelet
        Wavelet w(wavelet_name);

        // Perform multilevel decomposition
        auto coeffs = wavedecn(data, w, "symmetric", -1, {0});

        // Perform reconstruction
        auto rec = waverecn(coeffs, w, "symmetric", {0});

        // Compute error metrics
        double l2_error = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double diff = data.data()[i] - rec.data()[i];
            l2_error += diff * diff;
        }
        l2_error = std::sqrt(l2_error);

        std::cout << "Reconstruction L2 error: " << l2_error << std::endl;

        // Write reconstructed data
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        out.write(reinterpret_cast<const char*>(rec.data()), N * sizeof(double));
        out.close();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
