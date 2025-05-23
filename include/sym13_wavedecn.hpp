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

// --- Sym13 分解/重构滤波器（从 PyWavelets 提取） ---
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
        a[k] = ((k&1)? -1.0 : 1.0) * dec_lo[L-1-k];
    return a;
}();
static const std::array<double, L> rec_lo = {
  // rec_lo 从 PyWavelets 同步填入
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
        a[k] = (((k+1)&1)? -1.0 : 1.0) * rec_lo[L-1-k];
    return a;
}();

} // namespace sym13


// -------------------------
// 一个简易的多维数组类
// -------------------------
template<typename T>
class NDArray {
public:
    // **1. 默认构造：必须初始化 size_=0，否则 WaveCoeffs 默认构造时 cA 会无效**
    NDArray() : shape_(), strides_(), size_(0), data_() {}

    // 构造指定形状的数组，数据初始化为 0
    explicit NDArray(const std::vector<size_t>& shape)
        : shape_(shape)
    {
        size_ = 1;
        for (auto s : shape_) {
            if (s == 0) throw std::invalid_argument("NDArray: shape dimension cannot be zero");
            size_ *= s;
        }
        data_.assign(size_, T{});
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = (int)shape_.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    // **2. fromFlattened：不要再直接用 data_，而要用 data() 返回 T***
    static NDArray<T> fromFlattened(
        const std::vector<std::vector<T>>& mat,
        const std::vector<size_t>& shape)
    {
        int D = (int)shape.size();
        if (D < 1) throw std::invalid_argument("fromFlattened: need at least 1D");
        size_t rows = 1;
        for (int i = 0; i < D-1; ++i) rows *= shape[i];
        size_t cols = shape[D-1];
        if (mat.size() != rows)
            throw std::invalid_argument("fromFlattened: row count mismatch");
        NDArray<T> out(shape);
        T* ptr = out.data();            // <-- T* 指针
        for (size_t r = 0; r < rows; ++r) {
            if (mat[r].size() != cols)
                throw std::invalid_argument("fromFlattened: col count mismatch");
            std::copy(
                mat[r].begin(),
                mat[r].end(),
                ptr + r*cols        // <-- 用 ptr，而不是 out.data_
            );
        }
        return out;
    }

    // **3. reshape：同样不要传整个 vector，当 dest 迭代器，要用 data()**
    NDArray<T> reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = 1;
        for (auto s : new_shape) {
            if (s == 0) throw std::invalid_argument("reshape: zero dimension");
            new_size *= s;
        }
        if (new_size != size_)
            throw std::invalid_argument("reshape: total size mismatch");
        NDArray<T> out(new_shape);
        std::copy(
            data_.begin(),
            data_.end(),
            out.data()    // <-- T* dest
        );
        return out;
    }

    // **4. moveAxis：将某个轴 permute 到末尾之类的操作**
    NDArray<T> moveAxis(const std::vector<int>& perm) const {
        int D = (int)shape_.size();
        if ((int)perm.size() != D) throw std::invalid_argument("moveAxis: perm size mismatch");
        // 计算 new shape
        std::vector<size_t> new_shape(D);
        for (int i = 0; i < D; ++i) new_shape[i] = shape_[perm[i]];
        NDArray<T> out(new_shape);
        // 计算 new strides
        const auto& this_strides = strides_;
        auto out_strides = out.strides();
        std::vector<size_t> idx(D), old_idx(D);
        size_t total = out.size();
        for (size_t lin = 0; lin < total; ++lin) {
            size_t rem = lin;
            for (int d = 0; d < D; ++d) {
                idx[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            for (int d = 0; d < D; ++d) {
                old_idx[perm[d]] = idx[d];
            }
            // 计算在 this->data_ 上的偏移
            size_t off = 0;
            for (int d = 0; d < D; ++d) off += old_idx[d] * this_strides[d];
            out.data()[lin] = data_[off];
        }
        return out;
    }

    // 基本访问
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    const std::vector<size_t>& shape()   const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t size()  const { return size_; }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    std::vector<T> data_;
};


// -------------------------
// WaveCoeffs 存储结构
// -------------------------
template<typename T>
struct WaveCoeffs {
    // **5. 默认构造：要给 cA 一个合法的 shape**
    WaveCoeffs() : cA(std::vector<size_t>{1}), details(), original_shape() {}

    NDArray<T> cA;
    std::vector<std::map<std::string,NDArray<T>>> details;
    std::vector<size_t> original_shape;
};


// --------------------------------------
// pywt::dwt_max_level 的简易移植
// --------------------------------------
inline size_t dwt_max_level(size_t data_len, size_t filter_len) {
    if (filter_len < 2 || data_len < filter_len) return 0;
    double ratio = double(data_len) / double(filter_len - 1);
    return size_t(std::floor(std::log(ratio) / std::log(2.0)));
}


// --------------------------------------
// 1D DWT
// --------------------------------------
template<typename T>
static void dwt1d(const std::vector<T>& x,
                  const std::string& mode,
                  std::vector<T>& cA,
                  std::vector<T>& cD)
{
    const int N = int(x.size()), L = sym13::L;
    if (mode == "symmetric") {
        int pad = L-1;
        std::vector<T> ext(N+2*pad);
        // 对称延拓
        for (int i = 0; i < pad; ++i) {
            int idx = std::clamp(pad-1-i, 0, N-1);
            ext[i] = x[idx];
        }
        std::copy(x.begin(), x.end(), ext.begin()+pad);
        for (int i = 0; i < pad; ++i) {
            int idx = std::clamp(N-1-i, 0, N-1);
            ext[N+pad+i] = x[idx];
        }
        int outLen = (N + L - 1)/2;
        cA.assign(outLen, T(0));
        cD.assign(outLen, T(0));
        for (int k = 0; k < outLen; ++k) {
            T a=0, d=0;
            int st = 2*k;
            for (int j=0;j<L;++j) {
                a += T(sym13::dec_lo[j]) * ext[st+j];
                d += T(sym13::dec_hi[j]) * ext[st+j];
            }
            cA[k] = a;
            cD[k] = d;
        }
    }
    else {
        // 周期延拓
        int outLen = (N+1)/2;
        cA.assign(outLen, T(0));
        cD.assign(outLen, T(0));
        for (int k = 0; k < outLen; ++k) {
            for (int j = 0; j < L; ++j) {
                int idx = (2*k + j) % N;
                cA[k] += T(sym13::dec_lo[j]) * x[idx];
                cD[k] += T(sym13::dec_hi[j]) * x[idx];
            }
        }
    }
}


// --------------------------------------
// 1D IDWT
// --------------------------------------
template<typename T>
static void idwt1d(const std::vector<T>& cA,
                   const std::vector<T>& cD,
                   int N,
                   const std::string& mode,
                   std::vector<T>& out)
{
    int na = int(cA.size()), L = sym13::L;
    out.assign(N, T(0));
    if (mode == "symmetric") {
        int pad = L-1;
        std::vector<T> u(2*na), ext(2*na + 2*pad);
        for (int i=0;i<na;++i) {
            u[2*i]   = cA[i];
            u[2*i+1] = cD[i];
        }
        // 对称延拓
        for (int i=0;i<pad;++i) {
            int idx = std::clamp(pad-1-i, 0, 2*na-1);
            ext[i] = u[idx];
        }
        std::copy(u.begin(), u.end(), ext.begin()+pad);
        for (int i=0;i<pad;++i) {
            int idx = std::clamp(2*na-1-i, 0, 2*na-1);
            ext[2*na+pad+i] = u[idx];
        }
        // 长卷积
        for (int i=0;i<N;++i) {
            T v=0;
            for (int j=0;j<2*L;++j) {
                if ((j&1)==0)
                    v += T(sym13::rec_lo[j>>1]) * ext[i+j];
                else
                    v += T(sym13::rec_hi[j>>1]) * ext[i+j];
            }
            out[i] = v;
        }
    }
    else {
        // 周期
        for (int k=0;k<na;++k) {
            for (int j=0;j<2*L;++j) {
                int idx = (2*k + j) % N;
                if ((j&1)==0)
                    out[idx] += T(sym13::rec_lo[j>>1]) * cA[k];
                else
                    out[idx] += T(sym13::rec_hi[j>>1]) * cD[k];
            }
        }
    }
}


// --------------------------------------
// 单轴 DWT
// --------------------------------------
template<typename T>
static void dwt_axis(const NDArray<T>& arr,
                     int axis,
                     const std::string& mode,
                     NDArray<T>& outA,
                     NDArray<T>& outD)
{
    auto s = arr.shape(); int D = (int)s.size();
    // permute so that `axis` 在最后
    std::vector<int> perm(D); std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[axis], perm.back());
    auto moved = arr.moveAxis(perm);
    auto mshape = moved.shape();
    size_t M = 1;
    for (int i = 0; i < D-1; ++i) M *= mshape[i];
    size_t N = mshape.back();
    auto flat = moved.reshape({M,N});

    std::vector<T> row(N), cA, cD;
    std::vector<std::vector<T>> Adata, Ddata;
    Adata.reserve(M); Ddata.reserve(M);

    for (size_t i=0; i<M; ++i) {
        std::copy(flat.data()+i*N, flat.data()+(i+1)*N, row.begin());
        dwt1d(row, mode, cA, cD);
        Adata.push_back(cA);
        Ddata.push_back(cD);
    }
    // reshape 回新数组
    std::vector<size_t> sA(mshape.begin(), mshape.end()-1);
    sA.push_back(cA.size());
    std::vector<size_t> sD(mshape.begin(), mshape.end()-1);
    sD.push_back(cD.size());

    outA = NDArray<T>::fromFlattened(Adata, sA)
             .moveAxis([&]{
                 // inverse perm
                 std::vector<int> inv(D);
                 for (int i=0;i<D;++i) inv[perm[i]] = i;
                 return inv;
             }());
    outD = NDArray<T>::fromFlattened(Ddata, sD)
             .moveAxis([&]{
                 std::vector<int> inv(D);
                 for (int i=0;i<D;++i) inv[perm[i]] = i;
                 return inv;
             }());
}


// --------------------------------------
// 单轴逆变换 IDWT
// --------------------------------------
template<typename T>
static void idwt_axis(const NDArray<T>& arrA,
                      const NDArray<T>& arrD,
                      int axis_len,
                      const std::string& mode,
                      NDArray<T>& out)
{
    auto s = arrA.shape(); int D = (int)s.size();
    std::vector<int> perm(D); std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[axis_len], perm.back());
    auto mA = arrA.moveAxis(perm), mD = arrD.moveAxis(perm);
    auto mshape = mA.shape();
    size_t M = 1;
    for (int i=0;i<D-1;++i) M *= mshape[i];
    int N = axis_len;
    size_t NA = mshape.back();

    auto fA = mA.reshape({M,NA}), fD = mD.reshape({M,NA});
    std::vector<T> rowA(NA), rowD(NA), rec(N);
    std::vector<std::vector<T>> Rdata;
    Rdata.reserve(M);

    for (size_t i=0;i<M;++i) {
        std::copy(fA.data()+i*NA, fA.data()+(i+1)*NA, rowA.begin());
        std::copy(fD.data()+i*NA, fD.data()+(i+1)*NA, rowD.begin());
        idwt1d(rowA,rowD,N,mode,rec);
        Rdata.push_back(rec);
    }
    std::vector<size_t> sR(mshape.begin(), mshape.end()-1);
    sR.push_back(size_t(N));
    out = NDArray<T>::fromFlattened(Rdata, sR)
              .moveAxis([&]{
                  std::vector<int> inv(D);
                  for (int i=0;i<D;++i) inv[perm[i]] = i;
                  return inv;
              }());
}


// --------------------------------------
// wavedecn_simple
// --------------------------------------
template<typename T>
inline WaveCoeffs<T> wavedecn_simple(const NDArray<T>& data,
                                     const std::string& mode = "symmetric",
                                     int level = -1)
{
    WaveCoeffs<T> wc;
    wc.original_shape = data.shape();
    int D = (int)wc.original_shape.size();
    if (level < 0) {
        size_t m = *std::min_element(wc.original_shape.begin(), wc.original_shape.end());
        level = int(dwt_max_level(m, sym13::L));
    }
    std::cout<<level<<std::endl;
    NDArray<T> a = data;
    for (int lev = 0; lev < level; ++lev) {
         std::cout<<lev<<std::endl;
        std::map<std::string,NDArray<T>> cur;
        cur[""] = a;
        for (int ax = 0; ax < D; ++ax) {
            std::map<std::string,NDArray<T>> next;
            for (auto& kv : cur) {
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
        a = std::move(cur[std::string(D,'a')]);
        cur.erase(std::string(D,'a'));
        wc.details.push_back(std::move(cur));
    }
    wc.cA = std::move(a);
    return wc;
}


// --------------------------------------
// waverecn_simple
// --------------------------------------
template<typename T>
inline NDArray<T> waverecn_simple(const WaveCoeffs<T>& wc,
                                  const std::string& mode="symmetric")
{
    // 从 cA 开始
    NDArray<T> a = wc.cA;
    int ndim = int(a.ndim());

    // 每一级反向重构
    for (int lvl = int(wc.details.size()) - 1; lvl >= 0; --lvl) {
        // 拷贝出当前级的 detail map
        auto cur = wc.details[lvl]; // key 长度 == ndim

        // 把 approximation cA 补进来，key="aaa...a"
        std::string keyA(ndim, 'a');
        cur.emplace(keyA, a);

        // 对每一个轴，逐一做 1D IDWT，将 key 的长度减 1
        for (int ax = 0; ax < ndim; ++ax) {
            std::map<std::string, NDArray<T>> next;
            for (auto& kv : cur) {
                // 只对当前 key 中在 ax 处为 'a' 的条目做重构
                const auto& key = kv.first;
                if (key[ax] != 'a') continue;

                // 构造对应的 detail key，将第 ax 位改为 'd'
                auto keyD = key;
                keyD[ax] = 'd';

                auto itD = cur.find(keyD);
                if (itD == cur.end()) {
                    throw std::runtime_error("Missing detail subband for key " + keyD);
                }

                // 进行 1D 重构
                NDArray<T> rec;
                // idwt_axis: (cA, cD, axis length, mode, out)
                idwt_axis<T>(
                  /*cA*/  kv.second,
                  /*cD*/  itD->second,
                  /*axis_len*/ int(wc.original_shape[ax]),
                  /*mode*/    mode,
                  /*out*/     rec
                );

                // 新的 key，要把这一维度字母删掉，剩下 ndim-1 长度
                std::string newkey = key.substr(0, ax) + key.substr(ax+1);
                next.emplace(std::move(newkey), std::move(rec));
            }
            cur.swap(next);
        }

        // 此时 cur 里应该只有一个条目，key==""，对应完整重构后的 data
        if (cur.size() != 1 || cur.begin()->first != "") {
            throw std::runtime_error("waverecn_simple: unexpected map size after level " + std::to_string(lvl));
        }
        a = std::move(cur.begin()->second);
    }

    return a;
}

