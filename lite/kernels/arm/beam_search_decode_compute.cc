// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/arm/beam_search_decode_compute.h"
#include <algorithm>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using LoDTensor = lite::Tensor;
using LoDTensorArray = std::vector<lite::Tensor>;

// all the lod have 2 levels.
// The first is source level, the second is sentence level.
// source level describe how many prefixes (branchs) for each source sentece
// (beam). sentence level describe how these candidates belong to the prefixes.
const size_t kSourceLevel = 0;
const size_t kSentenceLevel = 1;

template <typename T>
struct Sentence {
  std::vector<int64_t> word_ids;
  std::vector<T> scores;
};

template <typename T>
using SentenceVector = std::vector<Sentence<T>>;

template <typename T>
struct BeamSearchDecoder {
  BeamSearchDecoder(size_t beam_size, int end_id)
      : beam_size_(beam_size), end_id_(end_id) {}

  /**
   * convert the result sentence_vector for each source sentence into two
   * LodTensor.
   * One is all candidate sentences with word id, one is all candidate sentences
   * with word score.
   * Param:
   *  sentence_vector_list: sentence_vector for each source sentence.
   *  id_tensor: result LoDTensor for sentences of id.
   *  score_tensor: result LoDTensor for sentences of score.
   *  reverse: whether ids of sentence in sentence_vector_list is reversed
   *  sort_by_score: whether to sort hypotheses of each sentence by scores.
   */
  void ConvertSentenceVectorToLodTensor(
      std::vector<SentenceVector<T>> sentence_vector_list,
      LoDTensor* id_tensor,
      LoDTensor* score_tensor,
      bool reverse = true,
      bool sort_by_score = true) const {
    size_t src_num = sentence_vector_list.size();
    CHECK_GT(src_num, 0) << "src_num should not be 0";

    std::vector<uint64_t> source_level_lod = {0};
    std::vector<uint64_t> sentence_level_lod = {0};
    std::vector<int64_t> id_data;
    std::vector<T> score_data;

    for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
      if (sort_by_score) {
        sort(sentence_vector_list[src_idx].begin(),
             sentence_vector_list[src_idx].end(),
             [reverse](const Sentence<T>& a, const Sentence<T>& b) {
               if (reverse)
                 return a.scores.front() > b.scores.front();
               else
                 return a.scores.back() > b.scores.back();
             });
      }
      for (Sentence<T>& sentence : sentence_vector_list[src_idx]) {
        if (reverse) {
          id_data.insert(id_data.end(),
                         sentence.word_ids.rbegin(),
                         sentence.word_ids.rend());
          score_data.insert(score_data.end(),
                            sentence.scores.rbegin(),
                            sentence.scores.rend());
        } else {
          id_data.insert(id_data.end(),
                         sentence.word_ids.begin(),
                         sentence.word_ids.end());
          score_data.insert(
              score_data.end(), sentence.scores.begin(), sentence.scores.end());
        }

        sentence_level_lod.push_back(sentence_level_lod.back() +
                                     sentence.word_ids.size());
      }
      source_level_lod.push_back(source_level_lod.back() +
                                 sentence_vector_list[src_idx].size());
    }

    LoD lod;
    lod.push_back(source_level_lod);
    lod.push_back(sentence_level_lod);

    *(id_tensor->mutable_lod()) = lod;

    id_tensor->Resize({static_cast<int64_t>(id_data.size())});
    auto id_ptr = id_tensor->mutable_data<int64_t>();
    TargetCopy(
        TARGET(kARM), id_ptr, id_data.data(), id_data.size() * sizeof(int64_t));

    *(score_tensor->mutable_lod()) = lod;
    score_tensor->Resize({static_cast<int64_t>(score_data.size())});
    auto score_ptr = score_tensor->mutable_data<T>();
    TargetCopy(TARGET(kARM),
               score_ptr,
               score_data.data(),
               score_data.size() * sizeof(T));
  }

  /**
   * Gather the hypotheses for each source sentence by backtrace though the
   * LoDTensorArray step_ids whose lods reserve the path in the tree.
   */
  void Backtrace(const LoDTensorArray& step_ids,
                 const LoDTensorArray& step_scores,
                 LoDTensor* id_tensor,
                 LoDTensor* score_tensor) const {
    CHECK(!step_ids.empty()) << "step num should be larger than 0";
    CHECK_EQ(step_ids.size(), step_scores.size())
        << "step_ids and step_scores should be the same";
    const size_t step_num = step_ids.size();
    const size_t src_num = step_ids.at(0).lod().at(kSourceLevel).size() - 1;
    std::vector<SentenceVector<T>> sentence_vector_list(
        src_num, SentenceVector<T>(beam_size_));
    std::vector<std::vector<size_t>> prefix_idx_vector_list(src_num);
    for (int step_id = step_num - 1; step_id >= 0; --step_id) {
      auto& cur_ids = step_ids.at(step_id);
      auto& cur_scores = step_scores.at(step_id);
      for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
        // for each source sentence
        auto& sentence_vector = sentence_vector_list.at(src_idx);
        auto& prefix_idx_vector = prefix_idx_vector_list.at(src_idx);
        size_t src_prefix_start = cur_ids.lod().at(kSourceLevel)[src_idx];
        size_t src_prefix_end = cur_ids.lod().at(kSourceLevel)[src_idx + 1];
        if (prefix_idx_vector.empty()) {  // be finished and pruned at this step
          // or the last time step
          for (size_t prefix_idx = src_prefix_start;
               prefix_idx < src_prefix_end;
               ++prefix_idx) {
            size_t candidate_start =
                cur_ids.lod().at(kSentenceLevel)[prefix_idx];
            size_t candidate_end =
                cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1];
            for (size_t candidate_idx = candidate_start;
                 candidate_idx < candidate_end;
                 ++candidate_idx) {
              prefix_idx_vector.push_back(prefix_idx);
              size_t idx = prefix_idx_vector.size() - 1;
              auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
              auto cur_score = cur_scores.data<T>()[candidate_idx];
              sentence_vector.at(idx).word_ids.push_back(cur_id);
              sentence_vector.at(idx).scores.push_back(cur_score);
            }
          }
        } else {  // use prefix_idx_vector to backtrace
          size_t src_candidate_start =
              cur_ids.lod().at(kSentenceLevel)[src_prefix_start];
          size_t prefix_idx = src_prefix_start;
          size_t candidate_num =
              cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1] -
              cur_ids.lod().at(kSentenceLevel)[prefix_idx];
          for (size_t idx = 0; idx < prefix_idx_vector.size(); ++idx) {
            auto candidate_idx = prefix_idx_vector.at(idx);
            auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
            auto cur_score = cur_scores.data<T>()[candidate_idx];
            if (cur_id != end_id_ || sentence_vector.at(idx).word_ids.empty()) {
              // to skip redundant end tokens
              sentence_vector.at(idx).word_ids.push_back(cur_id);
              sentence_vector.at(idx).scores.push_back(cur_score);
            }

            while (src_candidate_start + candidate_num <=
                   candidate_idx) {  // search the corresponding prefix
              prefix_idx++;
              candidate_num +=
                  cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1] -
                  cur_ids.lod().at(kSentenceLevel)[prefix_idx];
            }
            prefix_idx_vector.at(idx) = prefix_idx;
          }
        }
      }
    }

    ConvertSentenceVectorToLodTensor(
        sentence_vector_list, id_tensor, score_tensor, true, true);
  }

  size_t beam_size_;
  int end_id_;
};

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const LoDTensorArray& step_ids,
                          const LoDTensorArray& step_scores,
                          LoDTensor* id_tensor,
                          LoDTensor* score_tensor,
                          size_t beam_size,
                          int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        step_ids_(step_ids),
        step_scores_(step_scores),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {}

  template <typename T>
  void apply() const {
    BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
    beam_search_decoder.Backtrace(
        step_ids_, step_scores_, id_tensor_, score_tensor_);
  }

  size_t beam_size_;
  int end_id_;
  const LoDTensorArray& step_ids_;
  const LoDTensorArray& step_scores_;
  LoDTensor* id_tensor_;
  LoDTensor* score_tensor_;
};

template <>
void BeamSearchDecodeFunctor::apply<bool>() const {
  LOG(FATAL) << "beam search decode op does not support bool!";
}

void BeamSearchDecodeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  // inputs
  auto ids = param.ids;
  auto scores = param.scores;
  // outputs
  auto sentence_ids = param.sentence_ids;
  auto sentence_scores = param.sentence_scores;

  const size_t step_num = ids->size();
  CHECK_GT(step_num, 0UL) << "beam search steps should be larger than 0";
  const size_t source_num = ids->at(0).lod().at(0).size() - 1;
  CHECK_GT(source_num, 0UL) << "source num should be larger than 0";

  for (size_t i = 0; i < step_num; ++i) {
    CHECK_EQ(ids->at(i).lod().size(), 2UL) << "Level of LodTensor should be 2";
  }

  //! fixme
  // only support float score now
  BeamSearchDecodeFunctor func(*ids,
                               *scores,
                               sentence_ids,
                               sentence_scores,
                               param.beam_size,
                               param.end_id);

  func.apply<float>();

  // when decode finish, we clear ids and scores
  param.ids->clear();
  param.scores->clear();
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(beam_search_decode,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BeamSearchDecodeCompute,
                     def)
    .BindInput("Ids",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Scores",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("SentenceIds",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("SentenceScores",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
