#pragma once

#include <variant>
#include <map>
#include <vector>
#include <string>
#include "ndarray.hpp"
#include "wavelet.hpp"
#include "dwt.hpp"

// Type aliases for multilevel coefficients
template<typename T>
using DetailMap = std::map<std::string, NDArray<T>>;
template<typename T>
using CoeffList  = std::vector<std::variant<NDArray<T>, DetailMap<T>>>;

// Include full implementations
#include "multilevel_impl.hpp"
