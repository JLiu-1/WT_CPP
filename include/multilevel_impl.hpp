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

// Type aliases for multilevel coefficients
template<typename T>
using DetailMap = std::map<std::string, NDArray<T>>;
template<typename T>
using CoeffList = std::vector<std::variant<NDArray<T>, DetailMap<T>>>;

// --------------------------------------
// prep_axes_wavedecn
// --------------------------------------
inline std::tuple<std::vector<int>, std::vector<std::size_t>, int>
prep_axes_wavedecn(const std::vector<std::size_t>& shape,
                   const std::vector<int>& axes_in)
{
    if (shape.empty())
        throw std::invalid_argument("Expected at least 1D input data.");
    int ndim = static_cast<int>(shape.size());
    std::vector<int> axes = axes_in;
    if (axes.empty()) {
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }
    auto tmp = axes;
    std::sort(tmp.begin(), tmp.end());
    if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
        throw std::invalid_argument("Axes must be unique.");
    std::vector<std::size_t> axes_shapes;
    axes_shapes.reserve(axes.size());
    for (int ax : axes) {
        if (ax < 0 || ax >= ndim)
            throw std::out_of_range("Axis out of range");
        axes_shapes.push_back(shape[ax]);
    }
    return {axes, axes_shapes, static_cast<int>(axes.size())};
}

// --------------------------------------
// prepare_coeffs_axes
// --------------------------------------
template<typename T>
inline std::tuple<
    std::vector<std::variant<NDArray<T>,DetailMap<T>>>,
    std::vector<int>, int, int>
prepare_coeffs_axes(
    std::vector<std::variant<NDArray<T>,DetailMap<T>>> coeffs,
    std::vector<int> axes_in)
{
    if (coeffs.empty())
        throw std::invalid_argument("Empty coefficient list");
    int ndim = std::get<NDArray<T>>(coeffs[0]).ndim();
    int ndim_transform = 0;
    if (coeffs.size() > 1) {
        auto& d0 = std::get<DetailMap<T>>(coeffs[1]);
        if (d0.empty())
            throw std::invalid_argument("Invalid detail map");
        ndim_transform = static_cast<int>(d0.begin()->first.size());
    }
    auto axes = axes_in;
    if (axes.empty()) {
        if (ndim_transform < ndim)
            throw std::invalid_argument(
                "Axes must be specified for subset transforms.");
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }
    if (static_cast<int>(axes.size()) != ndim_transform)
        throw std::invalid_argument(
            "Length of axes must match number of dimensions transformed.");
    return {std::move(coeffs), axes, ndim, ndim_transform};
}

// --------------------------------------
// determine_coeff_array_shape
// --------------------------------------
template<typename T>
inline std::pair<std::vector<std::size_t>, bool>
determine_coeff_array_shape(
    const std::vector<std::variant<NDArray<T>,DetailMap<T>>>& coeffs,
    const std::vector<int>& axes)
{
    auto arr = std::get<NDArray<T>>(coeffs[0]);
    auto shape = arr.shape();
    std::size_t total = arr.size();
    int ndim_transform = static_cast<int>(axes.size());
    for (size_t i = 1; i < coeffs.size(); ++i) {
        auto& dmap = std::get<DetailMap<T>>(coeffs[i]);
        for (auto& kv : dmap)
            total += kv.second.size();
        auto& block = dmap.at(std::string(ndim_transform, 'd'));
        auto bshape = block.shape();
        for (int ax : axes)
            shape[ax] += bshape[ax];
    }
    std::size_t prod = 1;
    for (auto s : shape) prod *= s;
    return {shape, prod == total};
}

// --------------------------------------
// fix_coeffs
// --------------------------------------
template<typename T>
inline std::map<std::string, NDArray<T>>
fix_coeffs(const std::map<std::string, NDArray<T>>& coeffs)
{
    if (coeffs.empty())
        throw std::invalid_argument("Empty coefficient dictionary");
    size_t L = coeffs.begin()->first.size();
    for (auto& kv : coeffs) {
        if (kv.first.size() != L)
            throw std::invalid_argument("Inconsistent key lengths");
        for (char c : kv.first)
            if (c!='a' && c!='d')
                throw std::invalid_argument("Invalid coefficient key");
    }
    return coeffs;
}

// --------------------------------------
// Single-level n-D DWT: dwtn
// --------------------------------------
template<typename T>
inline std::map<std::string, NDArray<T>>
dwtn(const NDArray<T>& data,
     const Wavelet& w,
     const std::string& mode,
     const std::vector<int>& axes)
{
    int ndim = static_cast<int>(data.ndim());
    auto real_axes = axes;
    if (real_axes.empty()) {
        real_axes.resize(ndim);
        std::iota(real_axes.begin(), real_axes.end(), 0);
    }
    auto modes = modes_per_axis(mode, real_axes);
    auto wavelets = wavelets_per_axis(w, real_axes);
    int N = static_cast<int>(real_axes.size());
    std::vector<std::pair<std::string, NDArray<T>>> bands = {{"", data}};
    for (int i = 0; i < N; ++i) {
        int ax = real_axes[i];
        auto md = modes[i];
        auto& wav = wavelets[i];
        std::vector<std::pair<std::string, NDArray<T>>> next;
        for (auto& pr : bands) {
            auto [cA, cD] = dwt_axis(pr.second, wav, md, ax);
            next.emplace_back(pr.first + 'a', std::move(cA));
            next.emplace_back(pr.first + 'd', std::move(cD));
        }
        bands.swap(next);
    }
    std::map<std::string, NDArray<T>> result;
    for (auto& p : bands)
        result.emplace(p.first, std::move(p.second));
    return result;
}

// --------------------------------------
// Single-level n-D IDWT: idwtn
// --------------------------------------
template<typename T>
inline NDArray<T>
idwtn(std::map<std::string, NDArray<T>> coeffs,
      const Wavelet& w,
      const std::string& mode,
      const std::vector<int>& axes)
{
    auto cleaned = fix_coeffs(coeffs);
    int N = static_cast<int>(cleaned.begin()->first.size());
    auto real_axes = axes;
    if (real_axes.empty()) {
        real_axes.resize(N);
        std::iota(real_axes.begin(), real_axes.end(), 0);
    }
    auto modes = modes_per_axis(mode, real_axes);
    auto wavelets = wavelets_per_axis(w, real_axes);
    for (int lvl = N - 1; lvl >= 0; --lvl) {
        int ax = real_axes[lvl];
        auto md = modes[lvl];
        auto& wav = wavelets[lvl];
        std::map<std::string, NDArray<T>> next;
        std::string prefix;
        // Generate keys of length lvl
        for (auto& [k, _] : cleaned) {
            if ((int)k.size() != lvl + 1) continue;
            auto kA = k.substr(0, lvl) + 'a';
            auto kD = k.substr(0, lvl) + 'd';
            if (cleaned.count(kA) && cleaned.count(kD)) {
                next[k.substr(0, lvl)] = idwt_axis(cleaned[kA], cleaned[kD], wav, md, ax);
            }
        }
        cleaned.swap(next);
    }
    return cleaned[""];
}

// --------------------------------------
// Multilevel decomposition: wavedecn
// --------------------------------------
template<typename T>
inline CoeffList<T>
wavedecn(const NDArray<T>& data,
         const Wavelet& w,
         const std::string& mode,
         int level,
         const std::vector<int>& axes)
{
    auto [real_axes, axes_shapes, N] = prep_axes_wavedecn(data.shape(), axes);
    auto wavelets = wavelets_per_axis(w, real_axes);
    int max_level = (level < 0 ? static_cast<int>(axes_shapes.size()) : level);
    std::vector<DetailMap<T>> details;
    NDArray<T> approx = data;
    for (int i = 0; i < max_level; ++i) {
        auto coeffs = dwtn(approx, w, mode, real_axes);
        std::string akey(N, 'a');
        approx = coeffs.at(akey);
        coeffs.erase(akey);
        details.push_back(std::move(coeffs));
    }
    CoeffList<T> out;
    out.emplace_back(approx);
    for (auto& d : details)
        out.emplace_back(d);
    std::reverse(out.begin(), out.end());
    return out;
}

// --------------------------------------
// Multilevel reconstruction: waverecn
// --------------------------------------
template<typename T>
inline NDArray<T>
waverecn(const CoeffList<T>& coeffs,
          const Wavelet& w,
          const std::string& mode,
          const std::vector<int>& axes)
{
    if (coeffs.empty())
        throw std::invalid_argument("Coefficient list must have at least one element");
    std::vector<DetailMap<T>> details;
    for (size_t i = 1; i < coeffs.size(); ++i)
        details.push_back(std::get<DetailMap<T>>(coeffs[i]));
    NDArray<T> approx = std::get<NDArray<T>>(coeffs[0]);
    int levels = static_cast<int>(details.size());
    auto real_axes = axes;
    if (real_axes.empty()) {
        real_axes.resize(levels);
        std::iota(real_axes.begin(), real_axes.end(), 0);
    }
    for (int i = 0; i < levels; ++i) {
        auto& dmap = details[i];
        std::string akey(levels, 'a');
        dmap[akey] = approx;
        approx = idwtn(dmap, w, mode, real_axes);
    }
    return approx;
}
