#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "wavelet_registry.hpp"

// Forward C API types
struct DiscreteWaveletInternal;
std::pair<int,int> wname_to_code(const std::string&);
bool is_discrete_wav(int);
DiscreteWaveletInternal* discrete_wavelet(int, int);
DiscreteWaveletInternal* blank_discrete_wavelet(int);
void free_discrete_wavelet(DiscreteWaveletInternal*);

// Helper
inline void copy_c_array_to_vector(const double* src, int len, std::vector<double>& dst) {
    dst.assign(src, src+len);
}

class Wavelet {
public:
    explicit Wavelet(const std::string& name) : name_(to_lower(name)), internal_(nullptr) {
        auto code = wname_to_code(name_);
        if (!is_discrete_wav(code.first))
            throw std::invalid_argument("Wavelet '" + name_ + "' not discrete");
        internal_ = discrete_wavelet(code.first, code.second);
        if (!internal_) throw std::invalid_argument("Invalid wavelet name");
        dec_len_ = internal_->dec_len;
        rec_len_ = internal_->rec_len;
        copy_c_array_to_vector(internal_->dec_lo_double, dec_len_, dec_lo_);
        copy_c_array_to_vector(internal_->dec_hi_double, dec_len_, dec_hi_);
        copy_c_array_to_vector(internal_->rec_lo_double, rec_len_, rec_lo_);
        copy_c_array_to_vector(internal_->rec_hi_double, rec_len_, rec_hi_);
    }
    Wavelet(const std::string& name, const std::vector<std::vector<double>>& fb)
        : name_(name), internal_(nullptr) {
        if (fb.size()!=4) throw std::invalid_argument("filter_bank size!=4");
        dec_lo_=fb[0]; dec_hi_=fb[1]; rec_lo_=fb[2]; rec_hi_=fb[3];
        dec_len_ = dec_lo_.size(); rec_len_ = rec_lo_.size();
    }
    ~Wavelet() { if(internal_) free_discrete_wavelet(internal_); }

    int dec_len() const { return dec_len_; }
    int rec_len() const { return rec_len_; }
    const std::vector<double>& dec_lo() const { return dec_lo_; }
    const std::vector<double>& dec_hi() const { return dec_hi_; }
    const std::vector<double>& rec_lo() const { return rec_lo_; }
    const std::vector<double>& rec_hi() const { return rec_hi_; }

    std::vector<std::vector<double>> filter_bank() const {
        return {dec_lo_, dec_hi_, rec_lo_, rec_hi_};
    }
    std::vector<std::vector<double>> inverse_filter_bank() const {
        auto il=rec_lo_; std::reverse(il.begin(), il.end());
        auto ih=rec_hi_; std::reverse(ih.begin(), ih.end());
        auto dl=dec_lo_; std::reverse(dl.begin(), dl.end());
        auto dh=dec_hi_; std::reverse(dh.begin(), dh.end());
        return {il, ih, dl, dh};
    }

private:
    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    }
    std::string name_;
    DiscreteWaveletInternal* internal_;
    int dec_len_{0}, rec_len_{0};
    std::vector<double> dec_lo_, dec_hi_, rec_lo_, rec_hi_;
};

inline std::vector<Wavelet> wavelets_per_axis(
    const Wavelet& w, const std::vector<int>& axes) {
    return std::vector<Wavelet>(axes.size(), w);
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::string& name, const std::vector<int>& axes) {
    Wavelet w(name); return std::vector<Wavelet>(axes.size(), w);
}
inline std::vector<Wavelet> wavelets_per_axis(
    const std::vector<Wavelet>& wl, const std::vector<int>& axes) {
    if(wl.empty()) throw std::invalid_argument("Empty wavelet list");
    if(wl.size()==1) return std::vector<Wavelet>(axes.size(), wl[0]);
    if(wl.size()!=axes.size()) throw std::invalid_argument("Mismatched sizes");
    return wl;
}