#pragma once
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
    int ndim = data.ndim();
    auto shape = data.shape();
    if (axis < 0) axis += ndim;
    std::vector<std::size_t> new_shape = shape;
    new_shape[axis] += 2 * pad_width;
    NDArray<T> out(new_shape);
    auto ostr = out.strides();
    auto istr = data.strides();
    std::vector<std::size_t> idx(ndim), orig(ndim);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t r = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = r / ostr[d];
            r %= ostr[d];
        }
        int pos = int(idx[axis]) - pad_width;
        int L = int(shape[axis]);
        int p;
        if (mode == "symmetric") {
            if      (pos < 0)  p = -pos - 1;
            else if (pos >= L) p = 2*L - pos - 1;
            else               p = pos;
        } else if (mode == "periodic") {
            p = ((pos % L) + L) % L;
        } else if (mode == "constant") {
            p = std::clamp(pos, 0, L-1);
        } else if (mode == "zero") {
            if (pos < 0 || pos >= L) {
                out.data()[i] = T(0);
                continue;
            }
            p = pos;
        } else {
            throw std::invalid_argument("Unsupported pad mode");
        }
        for (int d = 0; d < ndim; ++d)
            orig[d] = (d == axis ? std::size_t(p) : idx[d]);
        auto off = std::inner_product(
            orig.begin(), orig.end(), istr.begin(), std::size_t(0));
        out.data()[i] = data.data()[off];
    }
    return out;
}

// --------------------------------------
// convolve_nd: 在轴上做一维卷积
// --------------------------------------
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
            idx[d] = r / ostr[d];
            r %= ostr[d];
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

// --------------------------------------
// downsample: 沿轴每2取1
// --------------------------------------
template<typename T>
NDArray<T> downsample(const NDArray<T>& data, int axis) {
    int ndim = data.ndim();
    auto shape = data.shape();
    std::vector<std::size_t> out_shape = shape;
    out_shape[axis] = (shape[axis] + 1) / 2;
    NDArray<T> out(out_shape);
    auto istr = data.strides();
    auto ostr = out.strides();
    std::vector<std::size_t> idx(ndim), in_idx(ndim);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t r = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = r / ostr[d];
            r %= ostr[d];
        }
        in_idx = idx;
        in_idx[axis] = idx[axis] * 2;
        auto off = std::inner_product(
            in_idx.begin(), in_idx.end(), istr.begin(), std::size_t(0));
        out.data()[i] = data.data()[off];
    }
    return out;
}

// --------------------------------------
// upsample: 沿轴插零
// --------------------------------------
template<typename T>
NDArray<T> upsample(const NDArray<T>& data, int axis) {
    int ndim = data.ndim();
    auto shape = data.shape();
    std::vector<std::size_t> out_shape = shape;
    out_shape[axis] = shape[axis] * 2;
    NDArray<T> out(out_shape);
    auto istr = data.strides();
    auto ostr = out.strides();
    std::vector<std::size_t> idx(ndim), in_idx(ndim);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t r = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = r / ostr[d];
            r %= ostr[d];
        }
        if (idx[axis] & 1) {
            out.data()[i] = T(0);
        } else {
            in_idx = idx;
            in_idx[axis] = idx[axis] / 2;
            auto off = std::inner_product(
                in_idx.begin(), in_idx.end(), istr.begin(), std::size_t(0));
            out.data()[i] = data.data()[off];
        }
    }
    return out;
}

// --------------------------------------
// trim_signal: 剪裁 pad 后多余
// --------------------------------------
template<typename T>
NDArray<T> trim_signal(const NDArray<T>& data,
                       int trim_width,
                       const std::string& /*mode*/,
                       int axis) {
    int ndim = data.ndim();
    auto shape = data.shape();
    std::vector<std::size_t> begin(ndim, 0), end = shape;
    end[axis] = shape[axis] - trim_width;
    NDArray<T> out(end);
    auto istr = data.strides();
    auto ostr = out.strides();
    std::vector<std::size_t> idx(ndim), in_idx(ndim);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t r = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = r / ostr[d];
            r %= ostr[d];
            in_idx[d] = idx[d] + begin[d];
        }
        auto off = std::inner_product(
            in_idx.begin(), in_idx.end(), istr.begin(), std::size_t(0));
        out.data()[i] = data.data()[off];
    }
    return out;
}

// --------------------------------------
// add_nd: 逐元素相加
// --------------------------------------
template<typename T>
NDArray<T> add_nd(const NDArray<T>& A, const NDArray<T>& B) {
    if (A.shape() != B.shape())
        throw std::invalid_argument("Shapes must match for add");
    NDArray<T> out(A.shape());
    for (std::size_t i = 0; i < A.size(); ++i)
        out.data()[i] = A.data()[i] + B.data()[i];
    return out;
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
    return { downsample(lo, axis), downsample(hi, axis) };
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
