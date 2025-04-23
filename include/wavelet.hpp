#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "wavelet_registry.hpp"

extern "C" {
#include "pywt_c/wavelets.h"
}

// Parse a string like "db1", "sym13", "bior6.8" → (WAVELET_NAME, order)
inline std::pair<WAVELET_NAME, unsigned int>
wname_to_code(const std::string& s) {
    std::string name = s;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (name == "haar"  || name == "db1") return {HAAR,  0};
    if (name.rfind("db", 0) == 0)   return {DB,   std::stoi(name.substr(2))};
    if (name.rfind("sym", 0) == 0)  return {SYM,  std::stoi(name.substr(3))};
    if (name.rfind("coif",0)== 0)  return {COIF, std::stoi(name.substr(4))};
    if (name.rfind("bior",0)== 0)  { auto p = name.find('.'); return {BIOR, std::stoi(name.substr(4, p-4))}; }
    if (name.rfind("rbio",0)== 0)  { auto p = name.find('.'); return {RBIO, std::stoi(name.substr(4, p-4))}; }
    if (name == "dmey")            return {DMEY,  0};
    if (name.rfind("gaus",0)== 0)  return {GAUS,  std::stoi(name.substr(4))};
    if (name == "mexh")            return {MEXH,  0};
    if (name == "morl")            return {MORL,  0};
    if (name.rfind("cgau",0)==0)   return {CGAU,  std::stoi(name.substr(4))};
    if (name.rfind("shan",0)==0)   return {SHAN,  std::stoi(name.substr(4))};
    if (name.rfind("fbsp",0)==0)   return {FBSP,  std::stoi(name.substr(4))};
    // Continuous wavelets not supported here
    throw std::invalid_argument("Unknown or continuous wavelet: " + s);
}

// Glue to the C API predicate
inline bool is_discrete_wav(WAVELET_NAME name) {
    return is_discrete_wavelet(name);
}

// Helper to copy C arrays to std::vector<double>
inline void copy_c_array_to_vector(const double* src, size_t len, std::vector<double>& dst) {
    dst.assign(src, src + len);
}

class Wavelet {
public:
    // Built-in discrete wavelet
    explicit Wavelet(const std::string& name)
      : name_(name)
      , w_(nullptr)
    {
        auto [fam, num] = wname_to_code(name_);
        if (!is_discrete_wav(fam))
            throw std::invalid_argument("Wavelet '" + name_ + "' is not discrete.");
        w_ = discrete_wavelet(fam, num);
        if (!w_)
            throw std::invalid_argument("Invalid wavelet name '" + name_ + "'.");
        dec_len_ = static_cast<int>(w_->dec_len);
        rec_len_ = static_cast<int>(w_->rec_len);
        copy_c_array_to_vector(w_->dec_lo_double, dec_len_, dec_lo_);
        copy_c_array_to_vector(w_->dec_hi_double, dec_len_, dec_hi_);
        copy_c_array_to_vector(w_->rec_lo_double, rec_len_, rec_lo_);
        copy_c_array_to_vector(w_->rec_hi_double, rec_len_, rec_hi_);
    }

    // Custom filter-bank wavelet
    Wavelet(const std::string& name,
            const std::vector<std::vector<double>>& filter_bank)
      : name_(name)
      , w_(nullptr)
    {
        if (filter_bank.size() != 4)
            throw std::invalid_argument("filter_bank must contain 4 filters.");
        dec_lo_ = filter_bank[0];
        dec_hi_ = filter_bank[1];
        rec_lo_ = filter_bank[2];
        rec_hi_ = filter_bank[3];
        dec_len_ = static_cast<int>(dec_lo_.size());
        rec_len_ = static_cast<int>(rec_lo_.size());
    }

    ~Wavelet() {
        if (w_)
            free_discrete_wavelet(w_);
    }

    // Accessors
    int dec_len() const { return dec_len_; }
    int rec_len() const { return rec_len_; }

    const std::vector<double>& dec_lo() const { return dec_lo_; }
    const std::vector<double>& dec_hi() const { return dec_hi_; }
    const std::vector<double>& rec_lo() const { return rec_lo_; }
    const std::vector<double>& rec_hi() const { return rec_hi_; }

    // Full banks
    std::vector<std::vector<double>> filter_bank() const {
        return { dec_lo_, dec_hi_, rec_lo_, rec_hi_ };
    }
    std::vector<std::vector<double>> inverse_filter_bank() const {
        auto il = rec_lo_; std::reverse(il.begin(), il.end());
        auto ih = rec_hi_; std::reverse(ih.begin(), ih.end());
        auto dl = dec_lo_; std::reverse(dl.begin(), dl.end());
        auto dh = dec_hi_; std::reverse(dh.begin(), dh.end());
        return { il, ih, dl, dh };
    }

private:
    std::string       name_;
    DiscreteWavelet*  w_;  // from wavelets.h

    int                         dec_len_{0}, rec_len_{0};
    std::vector<double>         dec_lo_, dec_hi_, rec_lo_, rec_hi_;
};

// wavelets_per_axis overloads
inline std::vector<Wavelet> wavelets_per_axis(
    const Wavelet& w, const std::vector<int>& axes)
{
    return std::vector<Wavelet>(axes.size(), w);
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::string& name, const std::vector<int>& axes)
{
    Wavelet w(name);
    return std::vector<Wavelet>(axes.size(), std::move(w));
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::vector<Wavelet>& wl, const std::vector<int>& axes)
{
    if (wl.empty())
        throw std::invalid_argument("wavelet list cannot be empty");
    if (wl.size() == 1)
        return std::vector<Wavelet>(axes.size(), wl[0]);
    if (wl.size() != axes.size())
        throw std::invalid_argument("Number of wavelets must match number of axes");
    return wl;
}