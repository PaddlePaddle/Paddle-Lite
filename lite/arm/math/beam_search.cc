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

#include "lite/arm/math/beam_search.h"
#include <arm_neon.h>
#include <cmath>
#include <string>
#include <vector>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
/*
* The basic items help to sort.
*/
struct Item {
  Item() {}
  Item(int offset, int64_t id, float score, int offset_in_alive)
      : offset(offset),
        id(id),
        score(score),
        offset_in_alive(offset_in_alive) {}
  Item(int offset, int64_t id, float score) : Item(offset, id, score, -1) {}

  // offset is the idx in pre_ids
  int offset;
  // the candidate id
  int64_t id;
  // the corresponding score
  float score;
  // Unused in beam search with the single queue. For beam search with alive
  // queue and finish queue, offset is the idx in pre_ids(including alive and
  // finish) and offset_in_alive is the idx in ids(including alive).
  int offset_in_alive;

  inline bool operator<(const Item &in) const {
    return (score < in.score) || ((score == in.score) && (offset < in.offset));
  }

  inline void operator=(const Item &in) {
    offset = in.offset;
    id = in.id;
    score = in.score;
    offset_in_alive = in.offset_in_alive;
  }

  std::string ToString() {
    std::ostringstream os;
    os << "{";
    os << "offset: " << offset << ", ";
    os << "id: " << id << ", ";
    os << "score: " << score << ", ";
    os << "offset_in_alive: " << offset_in_alive << "";
    os << "}";
    return os.str();
  }
};

static LoD ToAbsOffset(const LoD &in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
    }
  }
  return result;
}

/*
 * Prune the source sentences all branchs finished, and it is optional.
 * Pruning must one step later than finishing (thus pre_ids is needed here),
 * since the end tokens must be writed out.
 */
void PruneEndBeams(const Tensor *pre_ids,
                   const LoD &abs_lod,
                   std::vector<std::vector<Item>> *items,
                   size_t lod_level,
                   int end_id) {
  auto *pre_ids_data = pre_ids->data<float>();
  auto &high_level = abs_lod[lod_level];
  for (size_t src_idx = 0; src_idx < high_level.size() - 1; ++src_idx) {
    size_t src_prefix_start = high_level[src_idx];
    size_t src_prefix_end = high_level[src_idx + 1];
    bool finish_flag = true;
    for (size_t offset = src_prefix_start; offset < src_prefix_end; offset++) {
      for (auto &item : items->at(offset)) {
        if (item.id != static_cast<size_t>(end_id) ||
            pre_ids_data[offset] != end_id) {
          finish_flag = false;
          break;
        }
      }
      if (!finish_flag) break;
    }
    if (finish_flag) {  // all branchs of the beam (source sentence) end and
                        // prune this beam
      for (size_t offset = src_prefix_start; offset < src_prefix_end; offset++)
        items->at(offset).clear();
    }
  }
}

/*
 * Transform the items into a map whose key is offset, value is the items.
 * NOTE low performance.
 */
std::vector<std::vector<Item>> ToMap(
    const std::vector<std::vector<Item>> &items, size_t element_num) {
  std::vector<std::vector<Item>> result;
  result.resize(element_num);
  for (auto &entries : items) {
    for (const auto &item : entries) {
      result[item.offset].push_back(item);
    }
  }
  return result;
}

void Insert(std::vector<Item> *top_beam_ptr,
            const Item &item,
            size_t beam_size) {
  std::vector<Item> &top_beam = *top_beam_ptr;

  size_t num_beams = top_beam.size();
  if (num_beams < beam_size) {
    top_beam.resize(num_beams + 1);
    num_beams++;
  } else {
    if (item < top_beam[beam_size - 1]) {
      return;
    }
  }

  for (int k = static_cast<int>(num_beams) - 2; k >= 0; --k) {
    if (top_beam[k] < item) {
      top_beam[k + 1] = top_beam[k];
    } else {
      top_beam[k + 1] = item;
      return;
    }
  }
  top_beam[0] = item;
}

/*
 * For each source, select top beam_size records.
 */
std::vector<std::vector<Item>> SelectTopBeamSizeItems(const Tensor *pre_ids,
                                                      const Tensor *pre_scores,
                                                      const Tensor *ids,
                                                      const Tensor *scores,
                                                      size_t lod_level,
                                                      size_t beam_size,
                                                      int end_id,
                                                      bool is_accumulated) {
  std::vector<std::vector<Item>> result;

  // find the current candidates
  // auto abs_lod = framework::ToAbsOffset(scores->lod());
  auto abs_lod = scores->lod();
  auto *pre_ids_data = pre_ids->data<float>();
  auto *pre_scores_data = pre_scores->data<float>();

  auto *ids_data = ids ? ids->data<int>() : nullptr;
  auto *scores_data = scores->data<float>();

  size_t num_seqs = abs_lod[lod_level].size() - 1;
  size_t seq_width = 1;
  for (int i = 1; i < scores->dims().size(); i++) {
    seq_width *= scores->dims()[i];
  }

  for (size_t seq_id = 0; seq_id < num_seqs; ++seq_id) {
    size_t seq_offset_start = abs_lod[lod_level][seq_id];
    size_t seq_offset_end = abs_lod[lod_level][seq_id + 1];

    std::vector<Item> top_beam;
    top_beam.reserve(beam_size);

    for (size_t offset = seq_offset_start; offset < seq_offset_end; ++offset) {
      auto pre_id = pre_ids_data[offset];
      auto pre_score = pre_scores_data[offset];
      if (pre_id == end_id) {
        // Allocate all probability mass to end_id for finished branchs and
        // the other candidate ids can be ignored.
        Item item(offset, end_id, pre_score);
        Insert(&top_beam, item, beam_size);
      } else {
        size_t index = offset * seq_width;
        for (size_t d = 0; d < seq_width; d++, index++) {
          int64_t id = ids_data ? ids_data[index] : static_cast<int64_t>(d);
          float score = is_accumulated
                            ? scores_data[index]
                            : pre_score + std::log(scores_data[index]);
          Item item(offset, id, score);
          Insert(&top_beam, item, beam_size);
        }
      }
    }

    result.emplace_back(top_beam);
  }
  return result;
}
// for special implementation
static std::pair<int, int> SelectTopBeamSizeItems(
    const Tensor *pre_ids,
    const Tensor *pre_scores,
    const Tensor *ids,
    const Tensor *scores,
    const Tensor *src_length,
    const Tensor *cur_length,
    size_t level,
    size_t beam_size,
    int end_id,
    bool is_accumulated,
    int decode_length,
    float alpha,
    std::vector<std::vector<Item>> &alive_result,     // NOLINT
    std::vector<std::vector<Item>> &finish_result) {  // NOLINT
  // the lod of pre_scores have 2 levels.
  // The first level saves how many alive branchs for each source sentence.
  // The second level saves the idx in pre_ids rather than ids for the alive.
  auto &lod = pre_scores->lod();

  auto *pre_ids_data = pre_ids->data<float>();
  auto *pre_scores_data = pre_scores->data<float>();
  auto *ids_data = ids ? ids->data<int>() : nullptr;
  auto *scores_data = scores->data<float>();
  auto cur_len = cur_length->data<float>()[0];
  auto max_len = decode_length + (src_length ? src_length->dims()[1] : 0);

  float length_penalty = std::pow((5. + cur_len + 1) / 6., alpha);
  float pre_length_penalty = std::pow((5. + cur_len) / 6., alpha);

  size_t num_seqs = lod[level].size() - 1;
  size_t seq_width = 1;
  for (int i = 1; i < scores->dims().size(); i++) {
    seq_width *= scores->dims()[i];
  }

  int alive_num = 0;
  int finial_num = 0;
  int seq_id_in_finish = 0;  // differ to seq_id in alive
  // auto &lod_in_finsh = pre_ids->lod();  // differ to lod in alive
  auto lod_in_finsh = ToAbsOffset(pre_ids->lod());  // differ to lod in alive
  for (size_t seq_id = 0; seq_id < num_seqs; ++seq_id) {
    size_t seq_offset_start = lod[level][seq_id];
    size_t seq_offset_end = lod[level][seq_id + 1];

    if (seq_offset_start == seq_offset_end) {
      // have already finished, wonn't happen since we removed the finished
      // source sentence info in lod
      continue;
    }

    // grow_topk
    std::vector<Item> topk_top_beam;
    topk_top_beam.reserve(beam_size * 2);
    for (size_t offset = seq_offset_start; offset < seq_offset_end; ++offset) {
      size_t index = offset * seq_width;
      for (size_t d = 0; d < seq_width; d++, index++) {
        // offset(parent idx) should be idx in pre_ids or pre_scores,
        // which is saved in lod of pre_scores, rather than idx in scores
        auto pre_idx = lod[level + 1][offset];
        int64_t id = ids_data ? ids_data[index] : static_cast<int64_t>(d);
        // convert the length penalized score to log_prob
        float log_prob =
            (is_accumulated ? scores_data[index]
                            : pre_scores_data[pre_idx] * pre_length_penalty +
                                  std::log(scores_data[index]));
        float score = log_prob / length_penalty;
        Item item(pre_idx, id, score, offset);
        Insert(&topk_top_beam, item, beam_size * 2);
      }
    }

    // grow_alive and grow_finish
    std::vector<Item> alive_top_beam, finish_top_beam;
    alive_top_beam.reserve(beam_size);
    finish_top_beam.reserve(beam_size * 2);
    for (auto &item : topk_top_beam) {
      if (item.id == end_id) {
        // grow_finish
        finish_top_beam.emplace_back(std::move(item));
      } else if (alive_top_beam.size() < beam_size) {
        // grow_alive
        alive_top_beam.emplace_back(std::move(item));
      }
    }

    // find the range corresponding to current source in pre_ids
    auto idx_in_finish = alive_top_beam.front().offset;
    while (lod_in_finsh[level][seq_id_in_finish] <= idx_in_finish) {
      seq_id_in_finish++;
    }
    for (int i = lod_in_finsh[level][seq_id_in_finish - 1];
         i < lod_in_finsh[level][seq_id_in_finish];
         i++) {
      if (pre_ids_data[i] == end_id) {
        // grow_finish
        // pre_ids are not ordered by scores
        Item item(i, end_id, pre_scores_data[i]);
        Insert(&finish_top_beam, item, beam_size);
      }
    }

    // is_finish
    float max_length_penalty = std::pow((5. + max_len) / 6., alpha);
    if (cur_len == max_len ||
        (!finish_top_beam.empty() &&  // early stop
         alive_top_beam.front().score * length_penalty / max_length_penalty <
             finish_top_beam.back().score)) {
      // if finish, pad finish queue to beam_size and clear alive queue
      // should we clear alive queue if early stop?
      for (int i = 0; finish_top_beam.size() < beam_size; i++) {
        alive_top_beam[i].score = -INFINITY;
        finish_top_beam.emplace_back(std::move(alive_top_beam[i]));
      }
    } else {
      // if not finish, all alive items should be reserved to trace full path
      // thus alive is subset of finish, while the actual scores of alive
      // items in the finish queue should be -INF, put them at the back.
      finish_top_beam.insert(
          finish_top_beam.end(), alive_top_beam.begin(), alive_top_beam.end());
      alive_num += alive_top_beam.size();
      alive_result.emplace_back(std::move(alive_top_beam));
    }
    finial_num += finish_top_beam.size();
    finish_result.emplace_back(std::move(finish_top_beam));
  }

  return std::make_pair(alive_num, finial_num);  // NOLINT
}

void beam_search(const Tensor *pre_ids,
                 const Tensor *pre_scores,
                 const Tensor *ids,
                 const Tensor *scores,
                 Tensor *selected_ids,
                 Tensor *selected_scores,
                 Tensor *parent_idx,
                 int level,
                 int beam_size,
                 int end_id,
                 bool is_accumulated,
                 Context<TARGET(kARM)> *ctx) {
  // auto abs_lod = framework::ToAbsOffset(scores->lod());
  auto abs_lod = scores->lod();
  auto &high_level = abs_lod[level];
  auto items = SelectTopBeamSizeItems(pre_ids,
                                      pre_scores,
                                      ids,
                                      scores,
                                      level,
                                      beam_size,
                                      end_id,
                                      is_accumulated);
  auto selected_items = ToMap(items, high_level.back());

  PruneEndBeams(pre_ids, abs_lod, &selected_items, level, end_id);
  // calculate the output tensor's height
  size_t num_instances = std::accumulate(
      std::begin(selected_items),
      std::end(selected_items),
      0,
      [](size_t a, std::vector<Item> &b) { return a + b.size(); });
  // the output tensor shape should be [num_instances, 1]
  auto dims = std::vector<int64_t>({static_cast<int>(num_instances), 1});
  selected_ids->Resize(dims);
  selected_scores->Resize(dims);
  if (parent_idx) {
    parent_idx->Resize(dims);
  }
  auto *selected_ids_data = selected_ids->mutable_data<float>();
  auto *selected_scores_data = selected_scores->mutable_data<float>();
  auto *parent_idx_data =
      parent_idx ? parent_idx->mutable_data<int>() : nullptr;

  // fill in data
  std::vector<size_t> low_level;
  size_t low_offset = 0;
  for (auto &items : selected_items) {
    low_level.push_back(low_offset);
    for (auto &item : items) {
      if (parent_idx) {
        parent_idx_data[low_offset] = static_cast<int>(low_level.size() - 1);
      }
      selected_ids_data[low_offset] = item.id;
      selected_scores_data[low_offset] = item.score;
      low_offset++;
    }
  }
  low_level.push_back(low_offset);

  // fill lod
  LoD lod(2);
  lod[0].assign(high_level.begin(), high_level.end());
  lod[1].assign(low_level.begin(), low_level.end());
  *(selected_ids->mutable_lod()) = lod;
  *(selected_scores->mutable_lod()) = lod;
}

void beam_search_special(const Tensor *pre_ids,
                         const Tensor *pre_scores,
                         const Tensor *ids,
                         const Tensor *scores,
                         Tensor *selected_ids,
                         Tensor *selected_scores,
                         Tensor *parent_idx,
                         const Tensor *src_length,
                         const Tensor *cur_length,
                         Tensor *finish_scores,
                         Tensor *finish_ids,
                         int level,
                         int beam_size,
                         int end_id,
                         bool is_accumulated,
                         int decode_length,
                         float alpha,
                         Context<TARGET(kARM)> *ctx) {
  auto abs_lod = ToAbsOffset(pre_ids->lod());

  auto &high_level = abs_lod[level];
  std::vector<std::vector<Item>> alive_result, finish_result;
  // grow_topk, grow_alive, grow_finish
  auto num_instances = SelectTopBeamSizeItems(pre_ids,
                                              pre_scores,
                                              ids,
                                              scores,
                                              src_length,
                                              cur_length,
                                              level,
                                              beam_size,
                                              end_id,
                                              is_accumulated,
                                              decode_length,
                                              alpha,
                                              alive_result,
                                              finish_result);
  // map parent idx to corresponding selected(both alive and finish) items
  auto final_items = ToMap(finish_result, high_level.back());
  // calculate the output tensor's height
  auto alive_num = num_instances.first;
  auto finial_num = num_instances.second;

  auto alive_dims = std::vector<int64_t>({alive_num, 1, 1});
  auto finish_dims = std::vector<int64_t>({finial_num, 1});
  selected_ids->Resize(alive_dims);  // can resuse pre selected_ids
  selected_scores->Resize(alive_dims);
  finish_ids->Resize(finish_dims);  // can resuse pre finish_ids
  finish_scores->Resize(finish_dims);
  parent_idx->Resize({alive_num});

  auto *selected_ids_data = selected_ids->mutable_data<float>();
  auto *selected_scores_data = selected_scores->mutable_data<float>();
  auto *finish_ids_data = finish_ids->mutable_data<float>();
  auto *finish_scores_data = finish_scores->mutable_data<float>();
  auto *parent_idx_data = parent_idx->mutable_data<float>();

  // fill in data
  std::vector<size_t> low_level, alive_low_level;
  size_t low_offset = 0;
  size_t alive_offset = 0;
  for (auto &items : final_items) {
    low_level.push_back(low_offset);
    for (auto &item : items) {
      finish_ids_data[low_offset] = item.id;
      finish_scores_data[low_offset] = item.score;
      // we have make the alive subset of the finish
      if (item.id != end_id &&
          !std::isinf(item.score)) {  // otherwise, the item is used to pad
                                      // the finish queue to beam size when
                                      // finishing thus not alive
        selected_ids_data[alive_offset] = item.id;
        selected_scores_data[alive_offset] = item.score;
        parent_idx_data[alive_offset] = item.offset_in_alive;
        alive_low_level.push_back(low_offset);
        alive_offset++;
      }
      low_offset++;
    }
  }
  low_level.push_back(low_offset);

  // finish_lod has two level. The first is source level, the second is
  // sentence level. source level describe how many prefixes (branchs,
  // including alive and finish) for each source sentece (beam). sentence
  // level describe how these candidates belong to the prefixes.
  LoD finish_lod(2);
  finish_lod[0].assign(high_level.begin(), high_level.end());
  finish_lod[1].assign(low_level.begin(), low_level.end());
  // alive_lod have two levels. The first level saves how many alive branchs
  // for each source sentence. The second level saves the idx in pre_ids
  // rather than ids for the alive.
  LoD alive_lod(2);
  alive_lod[0].resize(alive_result.size() + 1);
  for (size_t i = 0; i <= alive_result.size(); i++) {
    alive_lod[0][i] = i * beam_size;
  }
  alive_lod[1].assign(alive_low_level.begin(), alive_low_level.end());

  // put alive_lod in finish_scores rather than selected_ids which wonn't be
  // the input of beam_search_op at next time step
  finish_ids->set_lod(finish_lod);
  finish_scores->set_lod(alive_lod);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
