#pragma once
#include <utility>
#include <type_traits>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "ndarray.hpp"
#include "wavelet.hpp"

// --------------------------------------
// modes_per_axis: 信号扩展模式映射
// --------------------------------------
inline std::vector<std::string> modes_per_axis(
    const std::string& mode,
    const std::vector<int>& axes) {
    return std::vector<std::string>(axes.size(), mode);
}
inline std::vector<std::string> modes_per_axis(
    const std::vector<std::string>& modes,
    const std::vector<int>& axes) {
    if (modes.size() == 1)
        return std::vector<std::string>(axes.size(), modes[0]);
    if (modes.size() != axes.size())
        throw std::invalid_argument(
            "Number of modes must match number of axes");
    return modes;
}

// --------------------------------------
// pad_signal: 按模式在轴两端填充
// --------------------------------------
template<typename T>
NDArray<T> pad_signal(const NDArray<T>& data,
                      int pad_width,
                      const std::string& mode,
                      int axis) {
    // ... (原有实现保持不变) ...
}

// --------------------------------------
// convolve_nd: 在轴上做一维卷积
// --------------------------------------
// 主模板：滤波器类型与数据类型相同
template<typename T>
NDArray<T> convolve_nd(const NDArray<T>& data,
                       const std::vector<T>& filter,
                       int axis) {
    int ndim = data.ndim();
    auto shape = data.shape();
    int L = int(filter.size());
    std::vector<std::size_t> out_shape = shape;
    out_shape[axis] = shape[axis] - (L - 1);
    NDArray<T> out(out_shape);
    auto istr = data.strides();
    auto ostr = out.strides();
    std::vector<std::size_t> idx(ndim), in_idx(ndim);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t r = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = r / ostr[d]; r %= ostr[d];
        }
        T sum = T(0);
        for (int k = 0; k < L; ++k) {
            in_idx = idx;
            in_idx[axis] = idx[axis] + k;
            auto off = std::inner_product(
                in_idx.begin(), in_idx.end(), istr.begin(), std::size_t(0));
            sum += filter[k] * data.data()[off];
        }
        out.data()[i] = sum;
    }
    return out;
}
// 重载：接受 double 滤波器，仅当 T != double
template<typename T, typename = std::enable_if_t<!std::is_same<T,double>::value>>
NDArray<T> convolve_nd(const NDArray<T>& data,
                       const std::vector<double>& filter,
                       int axis) {
    std::vector<T> f;
    f.reserve(filter.size());
    for (double v : filter) f.push_back(static_cast<T>(v));
    return convolve_nd<T>(data, f, axis);
}

// --------------------------------------
// downsample: 沿轴每2取1
// --------------------------------------
template<typename T>
NDArray<T> downsample(const NDArray<T>& data, int axis) {
    // ... (原有实现保持不变) ...
}

// --------------------------------------
// upsample: 沿轴插零
// --------------------------------------
template<typename T>
NDArray<T> upsample(const NDArray<T>& data, int axis) {
    // ... (原有实现保持不变) ...
}

// --------------------------------------
// trim_signal: 剪裁 pad 后多余
// --------------------------------------
template<typename T>
NDArray<T> trim_signal(const NDArray<T>& data,
                       int trim_width,
                       const std::string& /*mode*/,
                       int axis) {
    // ... (原有实现保持不变) ...
}

// --------------------------------------
// add_nd: 逐元素相加
// --------------------------------------
template<typename T>
NDArray<T> add_nd(const NDArray<T>& A, const NDArray<T>& B) {
    // ... (原有实现保持不变) ...
}

// --------------------------------------
// 单轴 DWT/IDWT
// --------------------------------------
template<typename T>
inline std::pair<NDArray<T>,NDArray<T>> dwt_axis(
    const NDArray<T>& data,
    const Wavelet& w,
    const std::string& mode,
    int axis) {
    auto p = pad_signal(data, w.dec_len()-1, mode, axis);
    auto lo = convolve_nd(p, w.dec_lo(), axis);
    auto hi = convolve_nd(p, w.dec_hi(), axis);
    return std::make_pair(downsample(lo, axis),
                          downsample(hi, axis));
}

template<typename T>
inline NDArray<T> idwt_axis(
    const NDArray<T>& cA,
    const NDArray<T>& cD,
    const Wavelet& w,
    const std::string& mode,
    int axis) {
    auto uA = upsample(cA, axis);
    auto uD = upsample(cD, axis);
    auto r = add_nd(convolve_nd(uA, w.rec_lo(), axis),
                    convolve_nd(uD, w.rec_hi(), axis));
    return trim_signal(r, w.rec_len()-1, mode, axis);
}