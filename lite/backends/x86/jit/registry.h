/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>  // for std::move
#include "lite/backends/x86/cpu_info.h"
#include "lite/backends/x86/jit/kernel_base.h"
#include "lite/backends/x86/jit/kernel_pool.h"
#include "lite/backends/x86/legacy_place.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace jit {

// make_unique is supported since c++14
template <typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
  static_assert(!std::is_array<T>::value, "T must not be array");
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename Pool,
          typename PlaceType,
          bool IsEnd,
          size_t I,
          typename... KernelImpls>
struct JitKernelRegistrarFunctor;

template <typename Pool, typename PlaceType, size_t I, typename... KernelImpls>
struct JitKernelRegistrarFunctor<Pool, PlaceType, true, I, KernelImpls...> {
  void operator()(KernelType kt) const {}
};

template <typename Pool, typename PlaceType, size_t I, typename... KernelImpls>
struct JitKernelRegistrarFunctor<Pool, PlaceType, false, I, KernelImpls...> {
  using KERNEL_IMPL_TYPE =
      typename std::tuple_element<I, std::tuple<KernelImpls...>>::type;

  void operator()(KernelType kt) const {
    KernelKey kkey(kt, PlaceType());
    Pool::Instance().Insert(kkey,
                            std::move(make_unique<const KERNEL_IMPL_TYPE>()));
    constexpr auto size = std::tuple_size<std::tuple<KernelImpls...>>::value;
    JitKernelRegistrarFunctor<Pool,
                              PlaceType,
                              I + 1 == size,
                              I + 1,
                              KernelImpls...>
        func;
    func(kt);
  }
};

template <typename Pool, typename PlaceType, typename... KernelImpls>
class JitKernelRegistrar {
 public:
  explicit JitKernelRegistrar(KernelType kt) {
    JitKernelRegistrarFunctor<Pool, PlaceType, false, 0, KernelImpls...> func;
    func(kt);
  }
  void Touch() {}
};

#define STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(uniq_name, msg)         \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// Refer always on CPUPlace
#define REGISTER_JITKERNEL_REFER_LITE(kernel_type, ...)             \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                    \
      __reg_litejitkernel_##kernel_type##_refer_CPUPlace,           \
      "REGISTER_KERNEL_REFER must be called in global namespace");  \
  static ::paddle::lite::jit::JitKernelRegistrar<                   \
      ::paddle::lite::jit::ReferKernelPool,                         \
      ::paddle::lite::fluid::CPUPlace,                              \
      __VA_ARGS__>                                                  \
      __jit_kernel_registrar_##kernel_type##_refer_CPUPlace_(       \
          ::paddle::lite::jit::KernelType::kernel_type);            \
  int LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_() {     \
    __jit_kernel_registrar_##kernel_type##_refer_CPUPlace_.Touch(); \
    return 0;                                                       \
  }

// kernel_type: should be in paddle::lite::jit::KernelType
// place_type: should be one of CPUPlace and GPUPlace in paddle::platform
#define REGISTER_KERNEL_MORE_LITE(kernel_type, impl_type, place_type, ...)    \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                              \
      __reg_litejitkernel_##kernel_type##_##impl_type##_##place_type,         \
      "REGISTER_KERNEL_MORE_LITE must be called in global namespace");        \
  extern int LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_();         \
  static int __assert_##kernel_type##_##impl_type##_##place_type##_has_refer_ \
      UNUSED = LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_();       \
  static ::paddle::lite::jit::JitKernelRegistrar<                             \
      ::paddle::lite::jit::KernelPool,                                        \
      ::paddle::lite::fluid::place_type,                                      \
      __VA_ARGS__>                                                            \
      __jit_kernel_registrar_##kernel_type##_##impl_type##_##place_type##_(   \
          ::paddle::lite::jit::KernelType::kernel_type);                      \
  int LiteTouchJitKernelReg_##kernel_type##_##impl_type##_##place_type##_() { \
    __jit_kernel_registrar_##kernel_type##_##impl_type##_##place_type##_      \
        .Touch();                                                             \
    return 0;                                                                 \
  }

#define REGISTER_JITKERNEL_MORE(kernel_type, impl_type, ...) \
  REGISTER_KERNEL_MORE_LITE(kernel_type, impl_type, CPUPlace, __VA_ARGS__)

#define REGISTER_GPUKERNEL_MORE_LITE(kernel_type, impl_type, ...) \
  REGISTER_KERNEL_MORE_LITE(kernel_type, impl_type, GPUPlace, __VA_ARGS__)

#define REGISTER_JITKERNEL_GEN_LITE(kernel_type, ...)                    \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                         \
      __reg_litejitkernel_gen_##kernel_type##_CPUPlace_,                 \
      "REGISTER_JITKERNEL_GEN_LITE must be called in global namespace"); \
  extern int LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_();    \
  static int __assert_gen_##kernel_type##_has_refer_ UNUSED =            \
      LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_();           \
  static ::paddle::lite::jit::JitKernelRegistrar<                        \
      ::paddle::lite::jit::JitCodeCreatorPool,                           \
      ::paddle::lite::fluid::CPUPlace,                                   \
      __VA_ARGS__>                                                       \
      __jit_kernel_registrar_gen_##kernel_type##_CPUPlace_(              \
          ::paddle::lite::jit::KernelType::kernel_type);                 \
  int LiteTouchJitKernelReg_gen_##kernel_type##_CPUPlace_() {            \
    __jit_kernel_registrar_gen_##kernel_type##_CPUPlace_.Touch();        \
    return 0;                                                            \
  }

#define USE_JITKERNEL_GEN_LITE(kernel_type)                           \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                      \
      __reg_litejitkernel_gen_##kernel_type##_CPUPlace_,              \
      "USE_JITKERNEL_GEN_LITE must be called in global namespace");   \
  extern int LiteTouchJitKernelReg_gen_##kernel_type##_CPUPlace_();   \
  static int use_litejitkernel_gen_##kernel_type##_CPUPlace_ UNUSED = \
      LiteTouchJitKernelReg_gen_##kernel_type##_CPUPlace_()

#define USE_JITKERNEL_REFER_LITE(kernel_type)                           \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                        \
      __reg_litejitkernel_##kernel_type##_refer_CPUPlace_,              \
      "USE_JITKERNEL_REFER_LITE must be called in global namespace");   \
  extern int LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_();   \
  static int use_litejitkernel_##kernel_type##_refer_CPUPlace_ UNUSED = \
      LiteTouchJitKernelReg_##kernel_type##_refer_CPUPlace_()

#define USE_KERNEL_MORE_LITE(kernel_type, impl_type, place_type)             \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                             \
      __reg_litejitkernel_##kernel_type##_##impl_type##_##place_type##_,     \
      "USE_JITKERNEL_MORE_LITE must be called in global namespace");         \
  extern int                                                                 \
      LiteTouchJitKernelReg_##kernel_type##_##impl_type##_##place_type##_(); \
  static int use_litejitkernel_##kernel_type##_##impl_type##_##place_type##_ \
      UNUSED =                                                               \
          LiteTouchJitKernelReg_##kernel_type##_##impl_type##_##place_type##_()

#define USE_JITKERNEL_MORE_LITE(kernel_type, impl_type) \
  USE_KERNEL_MORE_LITE(kernel_type, impl_type, CPUPlace)

}  // namespace jit
}  // namespace lite
}  // namespace paddle
