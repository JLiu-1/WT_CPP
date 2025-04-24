'''

import numpy as np
import pywt

# 从 PyWavelets 拿 sym13 系数
w = pywt.Wavelet('sym13')
dec_lo = np.array(w.dec_lo, dtype=float)
dec_hi = np.array(w.dec_hi, dtype=float)
rec_lo = np.array(w.rec_lo, dtype=float)
rec_hi = np.array(w.rec_hi, dtype=float)

L = dec_lo.size
pad = L - 1

def dwt1d(x, mode='symmetric'):
    x = np.asarray(x, dtype=float)
    N = x.size

    if mode == 'symmetric':
        # 对称延拓
        ext = np.pad(x, pad, mode='symmetric')
        # 全卷积
        y_lo = np.convolve(ext, dec_lo, mode='full')
        y_hi = np.convolve(ext, dec_hi, mode='full')
        # 取低频/高频子带
        na = (N + L - 1) // 2
        # pad+1 == L
        cA = y_lo[L : L + 2*na : 2]
        cD = y_hi[L : L + 2*na : 2]
        return cA, cD

    elif mode == 'periodic':
        # 周期延拓（periodization）：
        # cA[k] = sum_{t=0..L-1} dec_lo[t] * x[(2*k + L/2 - t) mod N]
        na = (N + 1) // 2
        cA = np.zeros(na, dtype=float)
        cD = np.zeros(na, dtype=float)
        off = L // 2
        for k in range(na):
            for t in range(L):
                idx = (2*k + off - t) % N
                cA[k] += dec_lo[t] * x[idx]
                cD[k] += dec_hi[t] * x[idx]
        return cA, cD

    else:
        raise ValueError(f"Unsupported mode '{mode}'")


def idwt1d(cA, cD, N, mode='symmetric'):
    cA = np.asarray(cA, dtype=float)
    cD = np.asarray(cD, dtype=float)
    na = cA.size

    if mode == 'symmetric':
        # 对称延拓重构
        # 先多相插值
        uA = np.zeros(2*na, dtype=float)
        uD = np.zeros(2*na, dtype=float)
        uA[::2] = cA
        uD[::2] = cD
        # 对称延拓
        extA = np.pad(uA, pad, mode='symmetric')
        extD = np.pad(uD, pad, mode='symmetric')
        # 有效卷积
        rA = np.convolve(extA, rec_lo, mode='valid')
        rD = np.convolve(extD, rec_hi, mode='valid')
        # 最佳切片位置 empirically: pad-1
        start = pad - 1
        return rA[start : start + N] + rD[start : start + N]

    elif mode == 'periodic':
        # 周期重构
        # out[n] += rec_lo_rev[t]*cA[k] + rec_hi_rev[t]*cD[k]
        out = np.zeros(N, dtype=float)
        off = L // 2
        # 需要用“反转”后的重构滤波器
        filt_lo = rec_lo[::-1]
        filt_hi = rec_hi[::-1]
        for k in range(na):
            for t in range(L):
                idx = (2*k + off - t) % N
                out[idx] += filt_lo[t] * cA[k] + filt_hi[t] * cD[k]
        return out

    else:
        raise ValueError(f"Unsupported mode '{mode}'")


# 测试
if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(200)

    for mode in ['symmetric', 'periodic']:
        py_mode = 'periodization' if mode == 'periodic' else mode

        # PyWavelets 参考
        cA_py, cD_py = pywt.dwt(x, 'sym13', mode=py_mode)
        rec_py      = pywt.idwt(cA_py, cD_py, 'sym13', mode=py_mode)

        # 我们的实现
        cA_us, cD_us = dwt1d(x, mode)
        rec_us       = idwt1d(cA_us, cD_us, x.size, mode)

        print(f"mode={mode}")
        print(f"  lenA_py={len(cA_py)} vs lenA_us={len(cA_us)}")
        print(f"  lenD_py={len(cD_py)} vs lenD_us={len(cD_us)}")
        print(f"  ΔcA   ={np.max(np.abs(cA_us - cA_py)):.2e}")
        print(f"  ΔcD   ={np.max(np.abs(cD_us - cD_py)):.2e}")
        print(f"  Δrec  ={np.max(np.abs(rec_us - rec_py)):.2e}")
        print()
'''

import numpy as np
import pywt

def wavedecn_simple(data, wavelet, mode='symmetric', level=None):
    data = np.asarray(data, dtype=float)
    original_shape = data.shape
    w = pywt.Wavelet(wavelet)
    dec_lo = np.array(w.dec_lo, dtype=float)
    dec_hi = np.array(w.dec_hi, dtype=float)
    L = dec_lo.size

    def dwt1d(x):
        N = x.size
        if mode == 'symmetric':
            pad = L - 1
            ext = np.pad(x, pad, mode='symmetric')
            y_lo = np.convolve(ext, dec_lo[::-1], mode='full')
            y_hi = np.convolve(ext, dec_hi[::-1], mode='full')
            na = (N + L - 1) // 2
            cA = y_lo[pad+1:pad+1+2*na:2]
            cD = y_hi[pad+1:pad+1+2*na:2]
        else:  # periodic
            na = (N + 1) // 2
            cA = np.zeros(na)
            cD = np.zeros(na)
            for k in range(na):
                for i in range(L):
                    idx = (2*k + i) % N
                    cA[k] += dec_lo[i] * x[idx]
                    cD[k] += dec_hi[i] * x[idx]
        return cA, cD

    def dwt_axis(arr, axis):
        arrm = np.moveaxis(arr, axis, -1)
        flat = arrm.reshape(-1, arrm.shape[-1])
        A = []; D = []
        for row in flat:
            ca, cd = dwt1d(row)
            A.append(ca); D.append(cd)
        A = np.array(A).reshape(arrm.shape[:-1] + (A[0].size,))
        D = np.array(D).reshape(arrm.shape[:-1] + (D[0].size,))
        return np.moveaxis(A, -1, axis), np.moveaxis(D, -1, axis)

    axes = list(range(data.ndim))
    if level is None:
        max_len = min(data.shape)
        level = pywt.dwt_max_level(max_len, L)

    coeffs = []
    a = data.copy()
    for _ in range(level):
        bands = {'': a}
        for ax in axes:
            new = {}
            for key, arr in bands.items():
                cA, cD = dwt_axis(arr, ax)
                new[key + 'a'] = cA
                new[key + 'd'] = cD
            bands = new
        a = bands.pop('a' * len(axes))
        coeffs.append(bands)
    coeffs.append(a)
    return coeffs[::-1], original_shape

def waverecn_simple(coeffs, wavelet, mode='symmetric', original_shape=None):
    w = pywt.Wavelet(wavelet)
    rec_lo = np.array(w.rec_lo, dtype=float)
    rec_hi = np.array(w.rec_hi, dtype=float)
    L = rec_lo.size

    def idwt1d(cA, cD, N):
        na = cA.size
        if mode == 'symmetric':
            pad = L - 1
            uA = np.zeros(2*na); uD = np.zeros(2*na)
            uA[::2] = cA; uD[::2] = cD
            extA = np.pad(uA, pad, mode='symmetric')
            extD = np.pad(uD, pad, mode='symmetric')
            rA = np.convolve(extA, rec_lo[::-1], mode='full')
            rD = np.convolve(extD, rec_hi[::-1], mode='full')
            return rA[pad:pad+N] + rD[pad:pad+N]
        else:  # periodic
            out = np.zeros(N)
            for k in range(na):
                for i in range(L):
                    idx = (2*k + i) % N
                    out[idx] += rec_lo[i]*cA[k] + rec_hi[i]*cD[k]
            return out

    def idwt_axis(cA, cD, axis):
        if mode == 'periodic':
            N = cA.shape[axis] + cD.shape[axis]
        else:
            if original_shape is None:
                raise ValueError("symmetric 模式需提供 original_shape")
            N = original_shape[axis]
        A = np.moveaxis(cA, axis, -1)
        D = np.moveaxis(cD, axis, -1)
        flatA = A.reshape(-1, A.shape[-1])
        flatD = D.reshape(-1, D.shape[-1])
        out_flat = [idwt1d(a, d, N) for a, d in zip(flatA, flatD)]
        out = np.array(out_flat).reshape(A.shape[:-1] + (N,))
        return np.moveaxis(out, -1, axis)

    a = coeffs[0]
    for lvl in range(1, len(coeffs)):
        det = coeffs[lvl]
        # 重建每层：先分解出轴上 cA/cD，然后沿各轴逆变换
        for ax in range(a.ndim):
            keyA = 'a'*ax + 'a' + 'a'*(a.ndim-ax-1)
            keyD = 'a'*ax + 'd' + 'a'*(a.ndim-ax-1)
            a = idwt_axis(det[keyA], det[keyD], ax)
    return a

if __name__ == '__main__':
    # 随机 3D 测试数据
    np.random.seed(0)
    data = np.random.randn(8,16,32)

    for mode in ['symmetric', 'periodic']:
        print(f"\n--- mode = {mode} ---")
        # PyWavelets 原生
        coeffs_py = pywt.wavedecn(data, 'sym13', mode=('periodization' if mode=='periodic' else mode))
        rec_py    = pywt.waverecn(coeffs_py, 'sym13', mode=('periodization' if mode=='periodic' else mode))
        # 简化版
        coeffs_us, orig_shape = wavedecn_simple(data, 'sym13', mode=mode, level=None)
        rec_us = waverecn_simple(coeffs_us, 'sym13', mode=mode, original_shape=orig_shape)

        # 比较顶层近似系数
        cA_py = coeffs_py[0]
        cA_us = coeffs_us[0]
        print("cA max abs diff:", np.max(np.abs(cA_py - cA_us)))
        # 比较各层细节
        for lvl in range(1, len(coeffs_py)):
            d_py = coeffs_py[lvl]
            d_us = coeffs_us[lvl]
            # 细节是 dict
            for key in d_py:
                diff = np.max(np.abs(d_py[key] - d_us[key]))
                print(f" lvl{lvl} band {key} diff {diff:.2e}")
        # 重构对比
        err_rec = np.max(np.abs(rec_py - rec_us))
        print("reconstruction max abs diff:", err_rec)
    