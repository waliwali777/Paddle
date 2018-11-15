/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <cmath>  // for exp
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"

#ifdef PADDLE_WITH_XBYAK
#include "paddle/fluid/operators/math/jit_code.h"
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace jit = platform::jit;

// TODO(TJ): move refer codes to one file
template <typename T>
void VExpRefer(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <typename T>
void VSigmoidRefer(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    T tmp = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-tmp));
  }
}

#ifdef PADDLE_WITH_MKLML
template <typename T>
void VExpMKL(const T* x, T* y, int n);

template <>
void VExpMKL<float>(const float* x, float* y, int n) {
  platform::dynload::vsExp(n, x, y);
}

template <>
void VExpMKL<double>(const double* x, double* y, int n) {
  platform::dynload::vdExp(n, x, y);
}

template <typename T>
void VSigmoidMKL(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  VExpMKL(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}
#endif

/* VExp JitKernel */
template <typename T>
class VExpKernelImpl : public VExpKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VExpKernelImpl(int d) : VExpKernel<T>() {
    this->num_ = d;  // TODO(TJ): remove me when ComputeDeprecated done
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / AVX_FLOAT_BLOCK * 4 * 8;  // should change
      jitcode_.reset(new gen::VExpJitCode(d, sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VExpMKL<T>;
      return;
    }
#endif
    this->Compute = VExpRefer<T>;
  }
  void ComputeDeprecated(const T* x, T* y) const override {
    VExpRefer(x, y, this->num_);
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VExpJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VExpKernelImpl<float>::useJIT(int d) {
  return gen::VExpJitCode::init(d);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VExpKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VExpKernelImpl<double>::useMKL(int d) {
  return true;
}

#endif

/* VSigmoid JitKernel */
template <typename T>
class VSigmoidKernelImpl : public VSigmoidKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VSigmoidKernelImpl(int d) : VSigmoidKernel<T>() {
    this->num_ = d;  // TODO(TJ): remove me when ComputeDeprecated done
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / AVX_FLOAT_BLOCK * 4 * 8;  // should change
      jitcode_.reset(new gen::VSigmoidJitCode(d, sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

#ifdef PADDLE_WITH_MKLML
    // strictly it's a better impl with MKL, then is refer
    if (useMKL(d)) {
      this->Compute = VSigmoidMKL<T>;
      return;
    }
#endif
    this->Compute = VSigmoidRefer<T>;
  }
  void ComputeDeprecated(const T* x, T* y) const override {
    VSigmoidRefer(x, y, this->num_);
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VSigmoidJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VSigmoidKernelImpl<float>::useJIT(int d) {
  return gen::VSigmoidJitCode::init(d);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VSigmoidKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VSigmoidKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

REGISTER_JITKERNEL(vexp, VExpKernel);
REGISTER_JITKERNEL(vsigmoid, VSigmoidKernel);

namespace detail {

#define ALIGN32 __attribute__((aligned(32)))

#define _PS256_CONST(Name, Val)                                      \
  static const float _ps256_##Name[8] ALIGN32 = {Val, Val, Val, Val, \
                                                 Val, Val, Val, Val}

#define _PI256_CONST(Name, Val)                                    \
  static const int _pi256_##Name[8] ALIGN32 = {Val, Val, Val, Val, \
                                               Val, Val, Val, Val}

_PI256_CONST(0x7f, 0x7f);
_PS256_CONST(one, 1.f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);
_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);
_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

typedef union imm_xmm_union {
  __m256i imm;
  __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_) \
  {                                         \
    imm_xmm_union u ALIGN32;                \
    u.imm = imm_;                           \
    xmm0_ = u.xmm[0];                       \
    xmm1_ = u.xmm[1];                       \
  }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_) \
  {                                         \
    imm_xmm_union u ALIGN32;                \
    u.xmm[0] = xmm0_;                       \
    u.xmm[1] = xmm1_;                       \
    imm_ = u.imm;                           \
  }

#define AVX2_BITOP_USING_SSE2(fn)                           \
  static inline __m256i avx2_mm256_##fn(__m256i x, int y) { \
    /* use SSE2 to perform the bitop AVX2 */                \
    __m128i x1, x2;                                         \
    __m256i ret;                                            \
    COPY_IMM_TO_XMM(x, x1, x2);                             \
    x1 = _mm_##fn(x1, y);                                   \
    x2 = _mm_##fn(x2, y);                                   \
    COPY_XMM_TO_IMM(x1, x2, ret);                           \
    return ret;                                             \
  }

#define AVX2_INTOP_USING_SSE2(fn)                                    \
  static inline __m256i avx2_mm256_add_epi32(__m256i x, __m256i y) { \
    /* use SSE2 to perform the AVX2 integer operation */             \
    __m128i x1, x2;                                                  \
    __m128i y1, y2;                                                  \
    __m256i ret;                                                     \
    COPY_IMM_TO_XMM(x, x1, x2);                                      \
    COPY_IMM_TO_XMM(y, y1, y2);                                      \
    x1 = _mm_##fn(x1, y1);                                           \
    x2 = _mm_##fn(x2, y2);                                           \
    COPY_XMM_TO_IMM(x1, x2, ret);                                    \
    return ret;                                                      \
  }

AVX2_BITOP_USING_SSE2(slli_epi32);
AVX2_INTOP_USING_SSE2(add_epi32);

#define AVXEXP_BASE                                                            \
  __m256 tmp = _mm256_setzero_ps(), fx;                                        \
  __m256 one = *reinterpret_cast<const __m256*>(_ps256_one);                   \
  __m256i imm0;                                                                \
  x = _mm256_min_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_hi));       \
  x = _mm256_max_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_lo));       \
  /* express exp(x) as exp(g + n*log(2)) */                                    \
  fx = _mm256_mul_ps(x,                                                        \
                     *reinterpret_cast<const __m256*>(_ps256_cephes_LOG2EF));  \
  fx = _mm256_add_ps(fx, *reinterpret_cast<const __m256*>(_ps256_0p5));        \
  tmp = _mm256_floor_ps(fx);                                                   \
  /* if greater, substract 1 */                                                \
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);                            \
  mask = _mm256_and_ps(mask, one);                                             \
  fx = _mm256_sub_ps(tmp, mask);                                               \
  tmp = _mm256_mul_ps(fx,                                                      \
                      *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C1)); \
  __m256 z = _mm256_mul_ps(                                                    \
      fx, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C2));             \
  x = _mm256_sub_ps(x, tmp);                                                   \
  x = _mm256_sub_ps(x, z);                                                     \
  z = _mm256_mul_ps(x, x);                                                     \
  __m256 y = *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p0);           \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p1));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p2));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p3));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p4));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p5));   \
  y = _mm256_mul_ps(y, z);                                                     \
  y = _mm256_add_ps(y, x);                                                     \
  y = _mm256_add_ps(y, one);                                                   \
  /* build 2^n */                                                              \
  imm0 = _mm256_cvttps_epi32(fx)

__m256 ExpAVX(__m256 x) {
  AVXEXP_BASE;
  // two AVX2 instructions using SSE2
  imm0 = avx2_mm256_add_epi32(imm0,
                              *reinterpret_cast<const __m256i*>(_pi256_0x7f));
  imm0 = avx2_mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

#ifdef __AVX2__
__m256 ExpAVX2(__m256 x) {
  AVXEXP_BASE;
  // two AVX2 instructions
  imm0 = _mm256_add_epi32(imm0, *reinterpret_cast<const __m256i*>(_pi256_0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
#endif

}  // namespace detail

#define INTRI_SIGMOID(tmp, min, max, expisa)      \
  tmp = _mm256_max_ps(tmp, min);                  \
  tmp = _mm256_min_ps(tmp, max);                  \
  tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp); \
  tmp = expisa(tmp);                              \
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp); \
  tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp)
#undef INTRI_VSIGMOID

/* VTanh JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class VTanhKernelImpl : public VTanhKernel<T> {
 public:
  explicit VTanhKernelImpl(int d) : VTanhKernel<T>() {
    this->num_ = d;
    vscal_ = KernelPool::Instance().template Get<VScalKernel<T>>(d);
    vsigmoid_ = KernelPool::Instance().template Get<VSigmoidKernel<T>>(d);
    vaddbias_ = KernelPool::Instance().template Get<VAddBiasKernel<T>>(d);
  }
  void ComputeDeprecated(const T* x, T* y) const override {
    const T a = static_cast<T>(2), b = static_cast<T>(-1);
    vscal_->Compute(&a, x, y, this->num_);
    vsigmoid_->ComputeDeprecated(y, y);
    vscal_->Compute(&a, y, y, this->num_);
    vaddbias_->Compute(&b, y, y, this->num_);
  }

 private:
  std::shared_ptr<const VScalKernel<T>> vscal_;
  std::shared_ptr<const VSigmoidKernel<T>> vsigmoid_;
  std::shared_ptr<const VAddBiasKernel<T>> vaddbias_;
};

#define INTRI_VTANH(tmp, expisa)                           \
  tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), tmp);         \
  tmp = _mm256_min_ps(tmp, _mm256_set1_ps(EXP_MAX_INPUT)); \
  tmp = expisa(tmp);                                       \
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);          \
  tmp = _mm256_div_ps(_mm256_set1_ps(2.0f), tmp);          \
  tmp = _mm256_sub_ps(tmp, _mm256_set1_ps(1.0f))

#define INTRI8_FLOAT(isa, expisa)                                             \
  template <>                                                                 \
  void VTanhKernelImpl<float, isa, kEQ8>::ComputeDeprecated(const float* x,   \
                                                            float* y) const { \
    __m256 tmp = _mm256_loadu_ps(x);                                          \
    INTRI_VTANH(tmp, expisa);                                                 \
    _mm256_storeu_ps(y, tmp);                                                 \
  }

#define INTRI16_FLOAT(isa, expisa)                                             \
  template <>                                                                  \
  void VTanhKernelImpl<float, isa, kEQ16>::ComputeDeprecated(const float* x,   \
                                                             float* y) const { \
    __m256 tmp0 = _mm256_loadu_ps(x);                                          \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);                                      \
    INTRI_VTANH(tmp0, expisa);                                                 \
    INTRI_VTANH(tmp1, expisa);                                                 \
    _mm256_storeu_ps(y, tmp0);                                                 \
    _mm256_storeu_ps(y + 8, tmp1);                                             \
  }

#define INTRI_GT8LT16_FLOAT(isa, expisa)                                      \
  template <>                                                                 \
  VTanhKernelImpl<float, isa, kGT8LT16>::VTanhKernelImpl(int d)               \
      : VTanhKernel<float>() {                                                \
    this->num_ = d;                                                           \
    this->end_ = AVX_FLOAT_BLOCK;                                             \
    this->rest_ = d - this->end_;                                             \
    vscal_ =                                                                  \
        KernelPool::Instance().template Get<VScalKernel<float>>(this->rest_); \
    vsigmoid_ = KernelPool::Instance().template Get<VSigmoidKernel<float>>(   \
        this->rest_);                                                         \
    vaddbias_ = KernelPool::Instance().template Get<VAddBiasKernel<float>>(   \
        this->rest_);                                                         \
  }                                                                           \
  template <>                                                                 \
  void VTanhKernelImpl<float, isa, kGT8LT16>::ComputeDeprecated(              \
      const float* x, float* y) const {                                       \
    __m256 tmp = _mm256_loadu_ps(x);                                          \
    INTRI_VTANH(tmp, expisa);                                                 \
    _mm256_storeu_ps(y, tmp);                                                 \
    x += AVX_FLOAT_BLOCK;                                                     \
    y += AVX_FLOAT_BLOCK;                                                     \
    const float a = 2.f, b = -1.f;                                            \
    vscal_->Compute(&a, x, y, this->num_);                                    \
    vsigmoid_->ComputeDeprecated(y, y);                                       \
    vscal_->Compute(&a, y, y, this->num_);                                    \
    vaddbias_->Compute(&b, y, y, this->num_);                                 \
  }

#define INTRI_GT16_FLOAT(isa, expisa)                                          \
  template <>                                                                  \
  VTanhKernelImpl<float, isa, kGT16>::VTanhKernelImpl(int d)                   \
      : VTanhKernel<float>() {                                                 \
    this->num_ = d;                                                            \
    this->rest_ = d % AVX_FLOAT_BLOCK;                                         \
    this->end_ = d - this->rest_;                                              \
    vscal_ =                                                                   \
        KernelPool::Instance().template Get<VScalKernel<float>>(this->rest_);  \
    vsigmoid_ = KernelPool::Instance().template Get<VSigmoidKernel<float>>(    \
        this->rest_);                                                          \
    vaddbias_ = KernelPool::Instance().template Get<VAddBiasKernel<float>>(    \
        this->rest_);                                                          \
  }                                                                            \
  template <>                                                                  \
  void VTanhKernelImpl<float, isa, kGT16>::ComputeDeprecated(const float* x,   \
                                                             float* y) const { \
    for (int i = 0; i < this->end_; i += AVX_FLOAT_BLOCK) {                    \
      __m256 tmp = _mm256_loadu_ps(x + i);                                     \
      INTRI_VTANH(tmp, expisa);                                                \
      _mm256_storeu_ps(y + i, tmp);                                            \
    }                                                                          \
    x += this->end_;                                                           \
    y += this->end_;                                                           \
    const float a = 2.f, b = -1.f;                                             \
    vscal_->Compute(&a, x, y, this->num_);                                     \
    vsigmoid_->ComputeDeprecated(y, y);                                        \
    vscal_->Compute(&a, y, y, this->num_);                                     \
    vaddbias_->Compute(&b, y, y, this->num_);                                  \
  }

#ifdef __AVX__
INTRI8_FLOAT(jit::avx, detail::ExpAVX);
INTRI16_FLOAT(jit::avx, detail::ExpAVX);
INTRI_GT8LT16_FLOAT(jit::avx, detail::ExpAVX);
INTRI_GT16_FLOAT(jit::avx, detail::ExpAVX);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2, detail::ExpAVX2);
INTRI16_FLOAT(jit::avx2, detail::ExpAVX2);
// maybe use avx at gt8lt16 and gt16
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f, detail::ExpAVX2);
INTRI16_FLOAT(jit::avx512f, detail::ExpAVX2);
// maybe use avx at gt8lt16 and gt16
#endif

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT
#undef INTRI_GT8LT16_FLOAT
#undef INTRI_GT16_FLOAT
#undef INTRI_VTANH

REGISTER_JITKERNEL_DEPRECATED(vtanh, VTanhKernel);

#undef JITKERNEL_NEW_ACT_IMPL

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
