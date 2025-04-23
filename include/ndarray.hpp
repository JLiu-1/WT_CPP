#pragma once

#include <vector>
#include <cstddef>
#include <numeric>
#include <algorithm>

// NDArray: N-dimensional array with contiguous storage
// Implements deep copy semantics via std::vector<T>

template <typename T>
class NDArray {
public:
    // Default constructor: empty array
    NDArray() = default;

    // Construct with given shape
    explicit NDArray(const std::vector<std::size_t>& shape)
        : shape_(shape) {
        // compute total size
        total_size_ = std::accumulate(
            shape_.begin(), shape_.end(), std::size_t(1), std::multiplies<>());
        // allocate storage
        data_.resize(total_size_);
        // compute strides (C-contiguous)
        strides_.resize(shape_.size());
        if (!shape_.empty()) {
            strides_.back() = 1;
            for (int i = int(shape_.size()) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }

    // Copy and move operators are defaulted: vector<T> handles deep copy
    NDArray(const NDArray&) = default;
    NDArray(NDArray&&) noexcept = default;
    NDArray& operator=(const NDArray&) = default;
    NDArray& operator=(NDArray&&) noexcept = default;

    // Accessors
    const std::vector<std::size_t>& shape() const { return shape_; }
    const std::vector<std::size_t>& strides() const { return strides_; }
    std::size_t size() const { return total_size_; }
    int ndim() const { return int(shape_.size()); }

    // Raw data pointer
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t total_size_{0};
    std::vector<T> data_;
};
