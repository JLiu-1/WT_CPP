// sym13_wavedecn.hpp
#pragma once

#include <vector>
#include <map>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

namespace sym13 {
// Sym13 分解/重构滤波器（从 PyWavelets 提取）
static constexpr int L = 26;
static const std::array<double, L> dec_lo = {
  7.0429866906944016e-005, 3.6905373423196241e-005, -0.0007213643851362283,
  0.00041326119884196064, 0.0056748537601224395, -0.0014924472742598532,
  -0.020749686325515677, 0.017618296880653084, 0.092926030899137119,
  0.0088197576704205465, -0.14049009311363403, 0.11023022302137217,
  0.64456438390118564, 0.69573915056149638, 0.19770481877117801,
  -0.12436246075153011, -0.059750627717943698, 0.013862497435849205,
  -0.017211642726299048, -0.02021676813338983, 0.0052963597387250252,
  0.0075262253899680996, -0.00017094285853022211, -0.0011360634389281183,
  -3.5738623648689009e-005, 6.8203252630753188e-005
};
static const std::array<double, L> dec_hi = []{
    std::array<double,L> a{};
    for(int k=0;k<L;++k)
        a[k] = ((k&1)? -1.0:1.0) * dec_lo[L-1-k];
    return a;
}();
static const std::array<double, L> rec_lo = {
  // 这里简化：rec_lo 用 PyWavelets 提供的值填入
  7.0429866906944016e-005, 3.6905373423196241e-005, -0.0007213643851362283,
  0.00041326119884196064, 0.0056748537601224395, -0.0014924472742598532,
  -0.020749686325515677, 0.017618296880653084, 0.092926030899137119,
  0.0088197576704205465, -0.14049009311363403, 0.11023022302137217,
  0.64456438390118564, 0.69573915056149638, 0.19770481877117801,
  -0.12436246075153011, -0.059750627717943698, 0.013862497435849205,
  -0.017211642726299048, -0.02021676813338983, 0.0052963597387250252,
  0.0075262253899680996, -0.00017094285853022211, -0.0011360634389281183,
  -3.5738623648689009e-005, 6.8203252630753188e-005
};
static const std::array<double, L> rec_hi = []{
    std::array<double,L> a{};
    for(int k=0;k<L;++k)
        a[k] = (((k+1)&1)? -1.0:1.0) * rec_lo[L-1-k];
    return a;
}();

} // namespace sym13

// WaveCoeffs 模板
template<typename T>
struct WaveCoeffs {
    NDArray<T> cA;
    std::vector<std::map<std::string,NDArray<T>>> details;
    std::vector<size_t> original_shape;
};

template<typename T>
class NDArray {
public:
    // 构造指定形状的数组，数据初始化为 0
    explicit NDArray(const std::vector<size_t>& shape)
        : shape_(shape)
    {
        // 计算总元素数
        size_ = 1;
        for (auto s : shape_) {
            if (s == 0) throw std::invalid_argument("NDArray: shape dimension cannot be zero");
            size_ *= s;
        }
        // 分配并置零
        data_.assign(size_, T{});
        // 计算 strides（行主序）
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = (int)shape_.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    // 元素访问
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // 属性访问
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t size()  const { return size_; }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    std::vector<T> data_;
};



// dwt_max_level 简单实现
inline size_t dwt_max_level(size_t data_len, size_t filter_len) {
    if (filter_len < 2 || data_len < filter_len) return 0;
    double ratio = double(data_len) / double(filter_len - 1);
    return size_t(std::floor(std::log(ratio)/std::log(2.0)));
}

// --------------------------------------
// 1D DWT 模板版
// --------------------------------------
template<typename T>
static void dwt1d(const std::vector<T>& x,
                  const std::string& mode,
                  std::vector<T>& cA,
                  std::vector<T>& cD)
{
    const int N = int(x.size());
    const int L  = sym13::L;
    if (mode == "symmetric") {
        int pad = L - 1;
        std::vector<T> ext(N + 2*pad);
        // 对称延拓
        for(int i=0;i<pad;++i){
            int idx = pad - 1 - i;
            idx = std::clamp(idx, 0, N-1);
            ext[i] = x[idx];
        }
        std::copy(x.begin(), x.end(), ext.begin()+pad);
        for(int i=0;i<pad;++i){
            int idx = N-1 - i;
            idx = std::clamp(idx, 0, N-1);
            ext[N+pad+i] = x[idx];
        }
        int outLen = (N + L - 1)/2;
        cA.assign(outLen, T(0));
        cD.assign(outLen, T(0));
        for(int k=0;k<outLen;++k){
            T a=0, d=0;
            int start = 2*k;
            for(int j=0;j<L;++j){
                a += T(sym13::dec_lo[j]) * ext[start+j];
                d += T(sym13::dec_hi[j]) * ext[start+j];
            }
            cA[k] = a;
            cD[k] = d;
        }
    } else {
        // periodic
        int outLen = (N+1)/2;
        cA.assign(outLen, T(0));
        cD.assign(outLen, T(0));
        for(int k=0;k<outLen;++k){
            for(int j=0;j<L;++j){
                int idx = (2*k + j) % N;
                cA[k] += T(sym13::dec_lo[j]) * x[idx];
                cD[k] += T(sym13::dec_hi[j]) * x[idx];
            }
        }
    }
}

// --------------------------------------
// 1D IDWT 模板版
// --------------------------------------
template<typename T>
static void idwt1d(const std::vector<T>& cA,
                   const std::vector<T>& cD,
                   int N,
                   const std::string& mode,
                   std::vector<T>& out)
{
    const int na = int(cA.size());
    const int L  = sym13::L;
    out.assign(N, T(0));
    if (mode == "symmetric") {
        int pad = L - 1;
        std::vector<T> u(2*na);
        for(int i=0;i<na;++i){
            u[2*i]   = cA[i];
            u[2*i+1] = cD[i];
        }
        std::vector<T> ext(2*na + 2*pad);
        // 对称延拓
        for(int i=0;i<pad;++i){
            int idx = pad - 1 - i;
            idx = std::clamp(idx, 0, 2*na-1);
            ext[i] = u[idx];
        }
        std::copy(u.begin(), u.end(), ext.begin()+pad);
        for(int i=0;i<pad;++i){
            int idx = 2*na-1 - i;
            idx = std::clamp(idx, 0, 2*na-1);
            ext[2*na+pad+i] = u[idx];
        }
        // 长卷积
        for(int i=0;i<N;++i){
            T v = 0;
            for(int j=0;j<2*L;++j){
                if ((j & 1) == 0)
                    v += T(sym13::rec_lo[j>>1]) * ext[i+j];
                else
                    v += T(sym13::rec_hi[j>>1]) * ext[i+j];
            }
            out[i] = v;
        }
    } else {
        // periodic
        for(int k=0;k<na;++k){
            for(int j=0;j<2*L;++j){
                int idx = (2*k + j) % N;
                if ((j & 1) == 0)
                    out[idx] += T(sym13::rec_lo[j>>1]) * cA[k];
                else
                    out[idx] += T(sym13::rec_hi[j>>1]) * cD[k];
            }
        }
    }
}

// --------------------------------------
// 单轴 DWT/IDWT（模板版）
// --------------------------------------
template<typename T>
static void dwt_axis(const NDArray<T>& arr,
                     int axis,
                     const std::string& mode,
                     NDArray<T>& outA,
                     NDArray<T>& outD)
{
    // 把 axis 维度移动到最后，然后当成 2D (M×N) 处理
    auto shape = arr.shape();
    int ndim = int(shape.size());
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[axis], perm.back());

    auto moved = arr.moveAxis(perm);
    auto ms = moved.shape();
    size_t M = 1;
    for(int i=0;i<ndim-1;++i) M *= ms[i];
    size_t N = ms.back();

    auto flat = moved.reshape({M, N});  // flatten to 2D
    std::vector<T> row(N), cA, cD;
    std::vector<std::vector<T>> Adata, Ddata;
    Adata.reserve(M); Ddata.reserve(M);

    for(size_t i=0;i<M;++i){
        std::copy(flat.data()+i*N, flat.data()+(i+1)*N, row.begin());
        dwt1d(row, mode, cA, cD);
        Adata.push_back(cA);
        Ddata.push_back(cD);
    }

    // 构建 outA/outD 的形状
    std::vector<size_t> sA(ms.begin(), ms.end()-1); sA.push_back(cA.size());
    std::vector<size_t> sD(ms.begin(), ms.end()-1); sD.push_back(cD.size());
    outA = NDArray<T>::fromFlattened(Adata, sA);
    outD = NDArray<T>::fromFlattened(Ddata, sD);

    // 再反向移动轴顺序
    std::vector<int> inv(ndim);
    for(int i=0;i<ndim;++i) inv[perm[i]] = i;
    outA = outA.moveAxis(inv);
    outD = outD.moveAxis(inv);
}

// --------------------------------------
// wavedecn_simple（模板版）
// --------------------------------------
template<typename T>
inline WaveCoeffs<T> wavedecn_simple(const NDArray<T>& data,
                                     const std::string& mode = "symmetric",
                                     int level = -1)
{
    WaveCoeffs<T> wc;
    wc.original_shape = data.shape();
    int ndim = int(wc.original_shape.size());
    std::vector<int> axes(ndim);
    std::iota(axes.begin(), axes.end(), 0);

    // 最高层数
    if(level < 0){
        size_t m = *std::min_element(wc.original_shape.begin(), wc.original_shape.end());
        level = int(dwt_max_level(m, sym13::L));
    }

    auto a = data;
    for(int lev=0; lev<level; ++lev){
        std::map<std::string,NDArray<T>> cur;
        cur[""] = a;
        for(int ax:axes){
            std::map<std::string,NDArray<T>> next;
            for(auto& kv:cur){
                auto key = kv.first;
                auto& arr = kv.second;
                NDArray<T> cA, cD;
                dwt_axis(arr, ax, mode, cA, cD);
                next[key+"a"] = std::move(cA);
                next[key+"d"] = std::move(cD);
            }
            cur.swap(next);
        }
        // 拿出纯 a
        a = std::move(cur[std::string(ndim,'a')]);
        cur.erase(std::string(ndim,'a'));
        wc.details.push_back(std::move(cur));
    }
    wc.cA = std::move(a);
    return wc;
}

// --------------------------------------
// waverecn_simple（模板版）
// --------------------------------------
template<typename T>
inline NDArray<T> waverecn_simple(const WaveCoeffs<T>& wc,
                                  const std::string& mode="symmetric")
{
    auto a = wc.cA;
    int ndim = int(a.ndim());
    for(int lev=int(wc.details.size())-1; lev>=0; --lev){
        const auto& det = wc.details[lev];
        for(int ax=0; ax<ndim; ++ax){
            std::string keyA(ndim,'a'), keyD(ndim,'a');
            keyA[ax] = 'a';
            keyD[ax] = 'd';
            auto cA = det.at(keyA);
            auto cD = det.at(keyD);
            // 重构当前维度
            NDArray<T> rec;
            // 注意：需要实现 idwt_axis<T>(...)
            idwt_axis(cA, cD, int(wc.original_shape[ax]), mode, rec);
            a = std::move(rec);
        }
    }
    return a;
}