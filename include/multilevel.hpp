#pragma once

#include <variant>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "ndarray.hpp"
#include "wavelet.hpp"
#include "dwt.hpp"

// 类型别名
template<typename T>
using DetailMap = std::map<std::string, NDArray<T>>;
template<typename T>
using CoeffList  = std::vector<std::variant<NDArray<T>, DetailMap<T>>>;

// 直接引入多级实现，不重复声明
#include "multilevel_impl.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "wavelet_registry.hpp"

extern "C" {
#include "pywt/_extensions/c/wavelets.h"
}

// Convert wavelet name string to (WAVELET_NAME, order)
inline std::pair<WAVELET_NAME,unsigned int>
wname_to_code(const std::string& s) {
    std::string name = s;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (name == "haar") {
        return {HAAR, 0};
    } else if (name.rfind("db", 0) == 0) {
        return {DB, static_cast<unsigned>(std::stoi(name.substr(2)))};
    } else if (name.rfind("sym", 0) == 0) {
        return {SYM, static_cast<unsigned>(std::stoi(name.substr(3)))};
    } else if (name.rfind("coif", 0) == 0) {
        return {COIF, static_cast<unsigned>(std::stoi(name.substr(4)))};
    } else if (name.rfind("bior", 0) == 0) {
        // biorthogonal: take integer part before dot
        auto pos = name.find('.');
        unsigned num = std::stoi(name.substr(4, pos-4));
        return {BIOR, num};
    } else if (name.rfind("rbio", 0) == 0) {
        auto pos = name.find('.');
        unsigned num = std::stoi(name.substr(4, pos-4));
        return {RBIO, num};
    } else if (name == "dmey") {
        return {DMEY, 0};
    } else if (name.rfind("gaus", 0) == 0) {
        return {GAUS, static_cast<unsigned>(std::stoi(name.substr(4)))};
    } else if (name == "mexh") {
        return {MEXH, 0};
    } else if (name == "morl") {
        return {MORL, 0};
    } else if (name.rfind("cgau", 0) == 0) {
        return {CGAU, static_cast<unsigned>(std::stoi(name.substr(4)))};
    } else if (name.rfind("shan", 0) == 0) {
        return {SHAN, static_cast<unsigned>(std::stoi(name.substr(4)))};
    } else if (name.rfind("fbsp", 0) == 0) {
        return {FBSP, static_cast<unsigned>(std::stoi(name.substr(4)))};
    } else if (name.rfind("cmor", 0) == 0) {
        // 'cmor' continuous wavelet not supported here
        throw std::invalid_argument("Continuous wavelet '" + s + "' not supported.");
    }
    throw std::invalid_argument("Unknown wavelet name: " + s);
}

inline bool is_discrete_wav_enum(WAVELET_NAME name) {
    return is_discrete_wavelet(name);
}

// Helper: copy C-array to std::vector<double>
inline void copy_c_array_to_vector(const double* src, size_t len, std::vector<double>& dst) {
    dst.assign(src, src + len);
}(const double* src, size_t len, std::vector<double>& dst) {
    dst.assign(src, src + len);
}

class Wavelet {
public:
    // Construct built-in discrete wavelet by name
    explicit Wavelet(const std::string& name)
      : name_(to_lower(name)), w_(nullptr) {
        auto code = wname_to_code(name_.c_str());
        if (!is_discrete_wav(code.first))
            throw std::invalid_argument("Wavelet '" + name_ + "' is not discrete");
        w_ = discrete_wavelet(code.first, code.second);
        if (!w_) throw std::invalid_argument("Invalid wavelet name '" + name_ + "'");
        dec_len_ = static_cast<int>(w_->dec_len);
        rec_len_ = static_cast<int>(w_->rec_len);
        copy_c_array_to_vector(w_->dec_lo_double, dec_len_, dec_lo_);
        copy_c_array_to_vector(w_->dec_hi_double, dec_len_, dec_hi_);
        copy_c_array_to_vector(w_->rec_lo_double, rec_len_, rec_lo_);
        copy_c_array_to_vector(w_->rec_hi_double, rec_len_, rec_hi_);
    }

    // Construct custom wavelet from explicit filter bank
    Wavelet(const std::string& name,
            const std::vector<std::vector<double>>& filter_bank)
      : name_(name), w_(nullptr) {
        if (filter_bank.size() != 4)
            throw std::invalid_argument("filter_bank must contain 4 filters");
        dec_lo_ = filter_bank[0];
        dec_hi_ = filter_bank[1];
        rec_lo_ = filter_bank[2];
        rec_hi_ = filter_bank[3];
        dec_len_ = static_cast<int>(dec_lo_.size());
        rec_len_ = static_cast<int>(rec_lo_.size());
    }

    ~Wavelet() {
        if (w_) free_discrete_wavelet(w_);
    }

    int dec_len() const { return dec_len_; }
    int rec_len() const { return rec_len_; }

    const std::vector<double>& dec_lo() const { return dec_lo_; }
    const std::vector<double>& dec_hi() const { return dec_hi_; }
    const std::vector<double>& rec_lo() const { return rec_lo_; }
    const std::vector<double>& rec_hi() const { return rec_hi_; }

    std::vector<std::vector<double>> filter_bank() const {
        return { dec_lo_, dec_hi_, rec_lo_, rec_hi_ };
    }
    std::vector<std::vector<double>> inverse_filter_bank() const {
        auto inv_lo = rec_lo_; std::reverse(inv_lo.begin(), inv_lo.end());
        auto inv_hi = rec_hi_; std::reverse(inv_hi.begin(), inv_hi.end());
        auto inv_dl = dec_lo_; std::reverse(inv_dl.begin(), inv_dl.end());
        auto inv_dh = dec_hi_; std::reverse(inv_dh.begin(), inv_dh.end());
        return { inv_lo, inv_hi, inv_dl, inv_dh };
    }

private:
    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    }

    std::string name_;
    DiscreteWavelet* w_;
    int dec_len_{0}, rec_len_{0};
    std::vector<double> dec_lo_, dec_hi_, rec_lo_, rec_hi_;
};

// wavelets_per_axis overloads
inline std::vector<Wavelet> wavelets_per_axis(
    const Wavelet& w, const std::vector<int>& axes) {
    return std::vector<Wavelet>(axes.size(), w);
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::string& name, const std::vector<int>& axes) {
    Wavelet w(name);
    return std::vector<Wavelet>(axes.size(), std::move(w));
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::vector<Wavelet>& wl, const std::vector<int>& axes) {
    if (wl.empty()) throw std::invalid_argument("wavelet list cannot be empty");
    if (wl.size() == 1)
        return std::vector<Wavelet>(axes.size(), wl[0]);
    if (wl.size() != axes.size())
        throw std::invalid_argument("Number of wavelets must match number of axes");
    return wl;
}
