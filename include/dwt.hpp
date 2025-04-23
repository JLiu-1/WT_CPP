#pragma once
#include <vector>
#include <string>
#include <numeric>
#include "ndarray.hpp"
#include "wavelet.hpp"


// Signal extension modes 对应 Python 的 _modes_per_axis
inline std::vector<std::string>
modes_per_axis(const std::string& mode, const std::vector<int>& axes) {
    return std::vector<std::string>(axes.size(), mode);
}


// pad_signal, convolve_nd, downsample, upsample, trim_signal, add_nd
// Refer to PyWavelets _extensions/_dwt.pyx

template<typename T>
NDArray<T> pad_signal(const NDArray<T>& data, int pad, const std::string& mode, int axis);

template<typename T>
NDArray<T> convolve_nd(const NDArray<T>& data, const std::vector<T>& filter, int axis);

template<typename T>
NDArray<T> downsample(const NDArray<T>& data, int axis);

template<typename T>
NDArray<T> upsample(const NDArray<T>& data, int axis);

template<typename T>
NDArray<T> trim_signal(const NDArray<T>& data, int trim_w, const std::string& mode, int axis);

template<typename T>
NDArray<T> add_nd(const NDArray<T>& A, const NDArray<T>& B);

template<typename T>
std::pair<NDArray<T>,NDArray<T>> dwt_axis(
    const NDArray<T>& data, const Wavelet& w, const std::string& mode, int axis) {
    auto p = pad_signal(data, w.dec_len()-1, mode, axis);
    auto cA = downsample(convolve_nd(p, w.dec_lo(), axis), axis);
    auto cD = downsample(convolve_nd(p, w.dec_hi(), axis), axis);
    return {cA, cD};
}

template<typename T>
NDArray<T> idwt_axis(
    const NDArray<T>& cA, const NDArray<T>& cD,
    const Wavelet& w, const std::string& mode, int axis) {
    auto uA = upsample(cA, axis);
    auto uD = upsample(cD, axis);
    auto r = add_nd(convolve_nd(uA, w.rec_lo(), axis),
                    convolve_nd(uD, w.rec_hi(), axis));
    return trim_signal(r, w.rec_len()-1, mode, axis);
}