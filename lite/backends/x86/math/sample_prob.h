/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <iostream>
#include <set>
#include <vector>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/math/sampler.h"
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

/* UNDERSTAND: utility function to adjust probability for unique sampling,
return whatever as it is if not using unique samping */
template <typename T>
static T adjust_prob(const T prob, const int num_samples, const int num_tries) {
  if (num_samples == num_tries) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

template <lite::TargetType Target, typename T>
class SampleWithProb {
 public:
  void operator()(const lite::Context<Target>& context,
                  const Sampler& sampler,
                  const std::size_t num_samples,
                  const lite::Tensor* L,
                  lite::Tensor* S,
                  lite::Tensor* P) {
    // UNDERSTAND: dimension issues
    const auto lbl_dim = L->dims();
    const int batch_size = lbl_dim[0];
    const int num_true = lbl_dim[1];
    const int num_sampled_classes = num_true + num_samples;
    // std::vector<int64_t> ret_dim_vec = {batch_size, num_sampled_classes};
    // lite::DDim ret_dim(ret_dim_vec);

    // UNDERSTAND: raw data view
    const int64_t* label_data = L->data<int64_t>();
    // int64_t* samples_data =
    //    S->mutable_data<int64_t>(ret_dim, Target);
    // T* probabilities_data = P->template mutable_data<T>(ret_dim, Target);
    S->Resize({batch_size, num_sampled_classes});
    auto* samples_data = S->mutable_data<int64_t>(Target);
    P->Resize({batch_size, num_sampled_classes});
    auto* probabilities_data = P->template mutable_data<T>(Target);

    // temp sets for unique sampling
    std::set<int64_t> tmp_samples;
    int j = 0;  // column index
    // add true labels, not that efficient
    while (j < num_true) {
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + j;
        auto v = label_data[i * num_true + j];
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = sampler.Probability(v);
      }
      ++j;
    }

    // sample num_samles unique samples for an example, note that they are not
    // all negative samples
    tmp_samples.clear();
    int num_tries = 0;
    while (j < num_sampled_classes) {
      ++num_tries;
      auto v = sampler.Sample();
      auto insert_ok = tmp_samples.insert(v).second;
      if (!insert_ok) {
        continue;
      }
      auto p = sampler.Probability(v);
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + j;
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = p;
      }
      ++j;
    }

    // compute Q(y|x), because of unique sampling, probabilities need to be
    // adjusted
    for (int k = 0; k < num_sampled_classes; ++k) {
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + k;
        probabilities_data[samples_index] = adjust_prob(
            probabilities_data[samples_index], num_samples, num_tries);
      }
    }
  }
};

// #ifdef PADDLE_WITH_CUDA
//  template <typename T>
//  class GPUSampleWithProb {
//  public:
//   void operator()(const platform::CUDAlite::Context<Target>& context, const
//   int seed,
//                   const int dict_size, const bool uniq,
//                   const std::size_t num_samples, const lite::Tensor* L,
//                   lite::Tensor* S,
//                   lite::Tensor* P);
// };
// #endif
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
