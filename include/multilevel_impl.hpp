#pragma once
#include "multilevel.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>

//
// Implementation for multilevel transforms
//

// prep_axes_wavedecn
inline std::tuple<std::vector<int>, std::vector<std::size_t>, int>
prep_axes_wavedecn(const std::vector<std::size_t>& shape,
                   const std::vector<int>& axes_in) {
    if (shape.empty())
        throw std::invalid_argument("Expected at least 1D input data.");
    int ndim = int(shape.size());
    std::vector<int> axes = axes_in;
    if (axes.empty()) {
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }
    {
        auto tmp = axes;
        std::sort(tmp.begin(), tmp.end());
        if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
            throw std::invalid_argument("Axes must be unique.");
    }
    std::vector<std::size_t> axes_shapes;
    axes_shapes.reserve(axes.size());
    for (int ax : axes) {
        if (ax < 0 || ax >= ndim)
            throw std::out_of_range("Axis out of range");
        axes_shapes.push_back(shape[ax]);
    }
    return {axes, axes_shapes, int(axes.size())};
}

// prepare_coeffs_axes
template<typename T>
inline std::tuple<
    std::vector<std::variant<NDArray<T>,DetailMap<T>>>,
    std::vector<int>, int, int>
prepare_coeffs_axes(
    std::vector<std::variant<NDArray<T>,DetailMap<T>>> coeffs,
    std::vector<int> axes_in) {
    if (coeffs.empty())
        throw std::invalid_argument("Empty coefficient list");
    int ndim = std::get<NDArray<T>>(coeffs[0]).ndim();
    int ndim_transform = 0;
    if (coeffs.size() > 1) {
        auto &d0 = std::get<DetailMap<T>>(coeffs[1]);
        if (d0.empty())
            throw std::invalid_argument("Invalid detail map");
        ndim_transform = int(d0.begin()->first.size());
    }
    auto axes = axes_in;
    if (axes.empty()) {
        if (ndim_transform < ndim)
            throw std::invalid_argument("Must specify axes for subset transform");
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }
    if (int(axes.size()) != ndim_transform)
        throw std::invalid_argument("Axes length mismatch");
    return {std::move(coeffs), axes, ndim, ndim_transform};
}

// determine_coeff_array_shape
template<typename T>
inline std::pair<std::vector<std::size_t>, bool>
determine_coeff_array_shape(
    const std::vector<std::variant<NDArray<T>,DetailMap<T>>>& coeffs,
    const std::vector<int>& axes) {
    auto arr = std::get<NDArray<T>>(coeffs[0]);
    auto out_shape = arr.shape();
    std::size_t total = arr.size();
    int ndim_transform = int(axes.size());
    for (size_t i = 1; i < coeffs.size(); ++i) {
        auto &dmap = std::get<DetailMap<T>>(coeffs[i]);
        for (auto &kv : dmap)
            total += kv.second.size();
        auto &block = dmap.at(std::string(ndim_transform,'d'));
        auto bshape = block.shape();
        for (int ax : axes)
            out_shape[ax] += bshape[ax];
    }
    std::size_t prod = 1;
    for (auto &s : out_shape) prod *= s;
    return {out_shape, prod == total};
}

// fix_coeffs
template<typename T>
inline std::map<std::string, NDArray<T>>
fix_coeffs(const std::map<std::string, NDArray<T>>& coeffs) {
    if (coeffs.empty())
        throw std::invalid_argument("Empty coeffs");
    size_t L = coeffs.begin()->first.size();
    for (auto &kv : coeffs) {
        auto &k = kv.first;
        if (k.size() != L)
            throw std::invalid_argument("Inconsistent key lengths");
        for (char c : k) if (c!='a' && c!='d')
            throw std::invalid_argument("Invalid key char");
    }
    return coeffs;
}

// dwtn
template<typename T>
inline std::map<std::string, NDArray<T>>
dwtn(const NDArray<T>& data, const Wavelet& w,
     const std::string& mode, const std::vector<int>& axes) {
    int ndim = data.ndim();
    auto real_axes = axes.empty() ? std::vector<int>(ndim) : axes;
    if (axes.empty()) std::iota(real_axes.begin(), real_axes.end(),0);
    auto modes = modes_per_axis(mode, real_axes);
    auto wvs = wavelets_per_axis(w, real_axes);
    int N = real_axes.size();
    std::vector<std::pair<std::string, NDArray<T>>> bands = {{"", data}};
    for (int i = 0; i < N; ++i) {
        int ax = real_axes[i];
        auto md = modes[i];
        auto &wv = wvs[i];
        std::vector<std::pair<std::string, NDArray<T>>> next;
        for (auto &pr : bands) {
            auto cA = dwt_axis(pr.second, wv, md, ax).first;
            auto cD = dwt_axis(pr.second, wv, md, ax).second;
            next.emplace_back(pr.first+'a', std::move(cA));
            next.emplace_back(pr.first+'d', std::move(cD));
        }
        bands.swap(next);
    }
    std::map<std::string, NDArray<T>> res;
    for (auto &pr : bands) res.emplace(pr.first, std::move(pr.second));
    return res;
}

// idwtn
template<typename T>
inline NDArray<T>
idwtn(std::map<std::string, NDArray<T>> coeffs,
      const Wavelet& w, const std::string& mode,
      const std::vector<int>& axes) {
    auto cleaned = fix_coeffs(coeffs);
    int N = cleaned.begin()->first.size();
    auto real_axes = axes.empty() ? std::vector<int>(N) : axes;
    if (axes.empty()) std::iota(real_axes.begin(), real_axes.end(),0);
    auto modes = modes_per_axis(mode, real_axes);
    auto wvs   = wavelets_per_axis(w, real_axes);
    for (int lvl = N-1; lvl >= 0; --lvl) {
        int ax = real_axes[lvl];
        auto md = modes[lvl];
        auto &wv = wvs[lvl];
        std::map<std::string, NDArray<T>> next;
        std::string prefix(N-lvl-1, 'a');
        for (auto &pr : cleaned) {
            auto &k = pr.first;
            if (k.size()==N) continue;
            auto kA = k+'a';
            auto kD = k+'d';
            if (cleaned.count(kA)&&cleaned.count(kD)) {
                next[k] = idwtn_axis(cleaned[kA], cleaned[kD], wv, md, ax);
            }
        }
        cleaned.swap(next);
    }
    return cleaned[""];
}

// wavedecn
template<typename T>
inline CoeffList<T> wavedecn(
    const NDArray<T>& data, const Wavelet& w,
    const std::string& mode, int level,
    const std::vector<int>& axes) {
    auto [ax, axsh, N] = prep_axes_wavedecn(data.shape(), axes);
    auto wvs = wavelets_per_axis(w, ax);
    std::vector<int> decs;
    for (auto &wv : wvs) decs.push_back(wv.dec_len());
    int maxlev = level<0? int(axsh.size()): level;
    std::vector<std::variant<NDArray<T>,DetailMap<T>>> out;
    NDArray<T> approx = data;
    std::vector<DetailMap<T>> dets;
    for (int i=0;i<maxlev;++i) {
        auto dmap = dwtn(approx, w, mode, ax);
        auto key = std::string(N,'a');
        approx = dmap.at(key);
        dmap.erase(key);
        dets.push_back(std::move(dmap));
    }
    out.emplace_back(approx);
    for (auto &d:dets) out.emplace_back(d);
    std::reverse(out.begin(), out.end());
    return out;
}

// waverecn
template<typename T>
inline NDArray<T> waverecn(
    const CoeffList<T>& coeffs, const Wavelet& w,
    const std::string& mode, const std::vector<int>& axes) {
    std::vector<DetailMap<T>> dets;
    for (size_t i = 1; i < coeffs.size(); ++i)
        dets.push_back(std::get<DetailMap<T>>(coeffs[i]));
    NDArray<T> approx = std::get<NDArray<T>>(coeffs.front());
    int N = dets.size();
    auto ax = axes;
    for (int i = 0; i < N; ++i) {
        auto &dmap = dets[i];
        auto key = std::string(N,'a');
        dmap[key] = approx;
        approx = idwtn(dmap, w, mode, ax);
    }
    return approx;
}

// coeffs_to_array and array_to_coeffs can be implemented similarly...
