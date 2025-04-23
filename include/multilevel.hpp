#pragma once
#include <vector>
#include <map>
#include <variant>
#include <algorithm>
#include "ndarray.hpp"
#include "wavelet.hpp"
#include "dwt.hpp"

// Type aliases
template<typename T>
using DetailMap = std::map<std::string, NDArray<T>>;
template<typename T>
using CoeffList = std::vector<std::variant<NDArray<T>, DetailMap<T>>>;

// Forward helpers (implementations in corresponding .cpp if needed)
std::tuple<std::vector<int>, std::vector<std::size_t>, int>
prep_axes_wavedecn(const std::vector<std::size_t>&, const std::vector<int>&);

template<typename T>
std::tuple<
    std::vector<std::variant<NDArray<T>,DetailMap<T>>>,
    std::vector<int>, int, int>
prepare_coeffs_axes(std::vector<std::variant<NDArray<T>,DetailMap<T>>>, std::vector<int>);

template<typename T>
std::pair<std::vector<std::size_t>,bool>
determine_coeff_array_shape(const std::vector<std::variant<NDArray<T>,DetailMap<T>>>&, const std::vector<int>&);

template<typename T>
std::map<std::string, NDArray<T>> fix_coeffs(const std::map<std::string, NDArray<T>>&);

// Core functions
template<typename T>
std::map<std::string, NDArray<T>> dwtn(
    const NDArray<T>& data, const Wavelet& w,
    const std::string& mode, const std::vector<int>& axes);

template<typename T>
NDArray<T> idwtn(
    std::map<std::string, NDArray<T>> coeffs, const Wavelet& w,
    const std::string& mode, const std::vector<int>& axes);

template<typename T>
CoeffList<T> wavedecn(
    const NDArray<T>& data, const Wavelet& w,
    const std::string& mode, int level, const std::vector<int>& axes);

template<typename T>
NDArray<T> waverecn(
    const CoeffList<T>& coeffs, const Wavelet& w,
    const std::string& mode, const std::vector<int>& axes);

template<typename T>
struct CoeffArrayResult { NDArray<T> arr; std::vector<std::map<std::string,std::pair<std::vector<std::size_t>,std::vector<std::size_t>>>> slices; };

template<typename T>
CoeffArrayResult<T> coeffs_to_array(
    std::vector<DetailMap<T>>, T padding, std::vector<int> axes);

template<typename T>
std::vector<DetailMap<T>> array_to_coeffs(
    const NDArray<T>& arr,
    const std::vector<std::map<std::string,std::pair<std::vector<std::size_t>,std::vector<std::size_t>>>>& slices,
    const std::string& format);
