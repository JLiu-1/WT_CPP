#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <type_traits>

template<typename T>
class NDArray {
public:

    NDArray() : ndim_(0), total_size_(0) {}

    
    explicit NDArray(const std::vector<std::size_t>& shape)
        : shape_(shape) {
        ndim_ = shape_.size();
        strides_.resize(ndim_);
        if (ndim_ > 0) {
            strides_.back() = 1;
            for (int i = ndim_-2; i >= 0; --i)
                strides_[i] = strides_[i+1] * shape_[i+1];
        }
        total_size_ = std::accumulate(
            shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());
        data_.resize(total_size_);
    }

    const std::vector<std::size_t>& shape()   const noexcept { return shape_; }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t ndim()                     const noexcept { return ndim_; }
    std::size_t size()                     const noexcept { return total_size_; }

    T*       data()       noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    template<typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(Args) == sizeof...(Args),
                      "Index count must match number of dimensions");
        std::array<std::size_t, sizeof...(Args)> idx{std::size_t(args)...};
        return data_[calc_offset(idx)];
    }
    template<typename... Args>
    const T& operator()(Args... args) const {
        std::array<std::size_t, sizeof...(Args)> idx{std::size_t(args)...};
        return data_[calc_offset(idx)];
    }

    T& at(const std::vector<std::size_t>& idx) {
        return data_[calc_offset(idx)];
    }
    const T& at(const std::vector<std::size_t>& idx) const {
        return data_[calc_offset(idx)];
    }

    void reshape(const std::vector<std::size_t>& new_shape) {
        std::size_t new_size = std::accumulate(
            new_shape.begin(), new_shape.end(), std::size_t{1}, std::multiplies<>());
        if (new_size != total_size_)
            throw std::invalid_argument("Total size must remain constant during reshape");
        shape_ = new_shape;
        ndim_ = shape_.size();
        strides_.resize(ndim_);
        if (ndim_ > 0) {
            strides_.back() = 1;
            for (int i = ndim_-2; i >= 0; --i)
                strides_[i] = strides_[i+1] * shape_[i+1];
        }
    }

    NDArray<T> slice(const std::vector<std::size_t>& begin,
                     const std::vector<std::size_t>& end) const {
        if (begin.size() != ndim_ || end.size() != ndim_)
            throw std::invalid_argument("Slice dimensions must match array dimensions");
        std::vector<std::size_t> new_shape(ndim_);
        for (size_t i = 0; i < ndim_; ++i) {
            if (end[i] <= begin[i] || end[i] > shape_[i])
                throw std::out_of_range("Invalid slice range");
            new_shape[i] = end[i] - begin[i];
        }
        NDArray<T> result(new_shape);
        for (std::size_t lin = 0; lin < result.total_size_; ++lin) {
            std::vector<std::size_t> idx(ndim_);
            std::size_t rem = lin;
            for (size_t d = 0; d < ndim_; ++d) {
                idx[d] = rem / result.strides_[d];
                rem %= result.strides_[d];
            }
            std::vector<std::size_t> orig(ndim_);
            for (size_t d = 0; d < ndim_; ++d)
                orig[d] = begin[d] + idx[d];
            result.data_[lin] = data_[calc_offset(orig)];
        }
        return result;
    }

    void slice_copy(const NDArray<T>& src,
                    const std::vector<std::size_t>& begin,
                    const std::vector<std::size_t>& end) {
        auto src_shape = src.shape();
        if (src_shape.size() != ndim_)
            throw std::invalid_argument("Source dimensions must match target");
        for (size_t i = 0; i < ndim_; ++i)
            if (end[i] - begin[i] != src_shape[i])
                throw std::invalid_argument("Slice size must match source");
        std::vector<std::size_t> idx(ndim_), tgt(ndim_);
        for (std::size_t lin = 0; lin < src.total_size_; ++lin) {
            std::size_t rem = lin;
            for (size_t d = 0; d < ndim_; ++d) {
                idx[d] = rem / src.strides_[d];
                rem %= src.strides_[d];
            }
            for (size_t d = 0; d < ndim_; ++d)
                tgt[d] = begin[d] + idx[d];
            data_[calc_offset(tgt)] = src.data_[lin];
        }
    }

    static NDArray<T> concatenate(const std::vector<NDArray<T>>& arrays,
                                  std::size_t axis) {
        if (arrays.empty()) throw std::invalid_argument("No arrays to concatenate");
        auto ref = arrays.front().shape();
        std::size_t ndim = ref.size();
        if (axis >= ndim) throw std::out_of_range("Axis out of range");
        for (auto& arr : arrays) {
            if (arr.shape().size() != ndim)
                throw std::invalid_argument("All arrays must have same dimensions");
            for (size_t d = 0; d < ndim; ++d)
                if (d != axis && arr.shape()[d] != ref[d])
                    throw std::invalid_argument("Non-concatenation dimensions must match");
        }
        std::vector<std::size_t> out_shape = ref;
        out_shape[axis] = 0;
        for (auto& arr : arrays) out_shape[axis] += arr.shape()[axis];
        NDArray<T> result(out_shape);
        std::size_t offset = 0;
        for (auto& arr : arrays) {
            std::vector<std::size_t> bgn(ndim), end = arr.shape();
            bgn[axis] = offset;
            for (size_t d = 0; d < ndim; ++d) end[d] += bgn[d];
            result.slice_copy(arr, bgn, end);
            offset += arr.shape()[axis];
        }
        return result;
    }

    static NDArray<T> broadcast_to(const NDArray<T>& arr,
                                   const std::vector<std::size_t>& new_shape) {
        std::size_t new_nd = new_shape.size();
        std::size_t old_nd = arr.ndim();
        if (old_nd > new_nd)
            throw std::invalid_argument("Cannot broadcast to fewer dimensions");
        std::vector<std::size_t> padded(new_nd, 1);
        for (size_t i = 0; i < old_nd; ++i)
            padded[new_nd-old_nd+i] = arr.shape()[i];
        for (size_t d = 0; d < new_nd; ++d)
            if (padded[d] != 1 && padded[d] != new_shape[d])
                throw std::invalid_argument("Shapes not broadcastable");
        NDArray<T> result(new_shape);
        std::vector<std::size_t> idx(new_nd), src_idx(old_nd);
        for (std::size_t lin = 0; lin < result.size(); ++lin) {
            std::size_t rem = lin;
            for (size_t d = 0; d < new_nd; ++d) {
                idx[d] = rem / result.strides_[d];
                rem %= result.strides_[d];
            }
            for (size_t i = 0; i < old_nd; ++i) {
                size_t d = new_nd-old_nd+i;
                src_idx[i] = (padded[d] == 1 ? 0 : idx[d]);
            }
            result.data_[lin] = arr.data_[std::inner_product(
                src_idx.begin(), src_idx.end(),
                arr.strides().begin(), std::size_t(0)
            )];
        }
        return result;
    }

private:
    template<typename C>
    std::size_t calc_offset(const C& idx) const {
        if (idx.size() != ndim_)
            throw std::out_of_range("Index rank mismatch");
        std::size_t off = 0;
        for (size_t i = 0; i < ndim_; ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("Index out of bounds");
            off += idx[i] * strides_[i];
        }
        return off;
    }

    std::vector<std::size_t> shape_, strides_;
    std::size_t ndim_{0}, total_size_{0};
    std::vector<T> data_;
};