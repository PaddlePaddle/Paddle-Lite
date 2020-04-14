/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/search_fc.h"
#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SearchFcFunctor<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& bottom,
                  const lite::Tensor& w,
                  const lite::Tensor& b,
                  lite::Tensor* top,
                  int out_size) {
    int batch = bottom.dims()[0];

    int _out = w.dims()[0];  // 100
    int _in = w.dims()[1];   // 228

    lite::DDim dims(std::vector<int64_t>({bottom.dims()[0], out_size}));

    const auto bottom_data = bottom.data<T>();
    auto top_data = top->template mutable_data<T>(lite::TargetType::kX86);
    const auto weights = w.data<T>();
    auto blas = math::GetBlas<lite::TargetType::kX86, T>(context);
    call_gemm<lite::X86Context, T>(blas,
                                   CblasNoTrans,
                                   CblasTrans,
                                   batch,
                                   _out,
                                   _in,
                                   1.0f,
                                   bottom_data,
                                   weights,
                                   0.0f,
                                   top_data);
    if (true) {
      const auto* bias_data = b.data<T>();
      for (int i = 0; i < batch; ++i) {
        // add bias here
        sse_eltadd(top_data + i * _out, bias_data, top_data + i * _out, _out);
      }
    }
  }

  // private:
};

#define DEFINE_FUNCTOR(type) \
  template class SearchFcFunctor<lite::TargetType::kX86, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
