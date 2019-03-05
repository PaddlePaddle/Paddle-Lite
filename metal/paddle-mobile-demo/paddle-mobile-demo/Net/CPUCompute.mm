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

#import "CPUCompute.h"

#import <map>
#import <vector>
#import <utility>
#import <algorithm>

struct NMSParam {
    
    float *score_data;
    
    float *box_data;
    
    float *output;
    
    int output_size;
    
    std::vector<int> score_dim;
    
    std::vector<int> box_dim;
    
    float scoreThredshold;
    
    int nmsTopK;
    
    int keepTopK;
    
    float nmsEta;
    
    float nmsThreshold;
    
    int background_label;
};


constexpr int kOutputDim = 6;
constexpr int kBBoxSize = 4;

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
                                    const std::vector<T>& scores, const T threshold, int top_k,
                                    std::vector<std::pair<T, int>>* sorted_indices) {
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            sorted_indices->push_back(std::make_pair(scores[i], i));
        }
    }
    // Sort the score pair according to the scores in descending order
    std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                     SortScorePairDescend<int>);
    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
        sorted_indices->resize(top_k);
    }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
    if (box[2] < box[0] || box[3] < box[1]) {
        // If coordinate values are is invalid
        // (e.g. xmax < xmin or ymax < ymin), return 0.
        return static_cast<T>(0.);
    } else {
        const T w = box[2] - box[0];
        const T h = box[3] - box[1];
        if (normalized) {
            return w * h;
        } else {
            // If coordinate values are not within range [0, 1].
            return (w + 1) * (h + 1);
        }
    }
}

template <class T>
static inline T JaccardOverlap(const T* box1, const T* box2,
                               const bool normalized) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
        box2[3] < box1[1]) {
        return static_cast<T>(0.);
    } else {
        const T inter_xmin = std::max(box1[0], box2[0]);
        const T inter_ymin = std::max(box1[1], box2[1]);
        const T inter_xmax = std::min(box1[2], box2[2]);
        const T inter_ymax = std::min(box1[3], box2[3]);
        const T inter_w = inter_xmax - inter_xmin;
        const T inter_h = inter_ymax - inter_ymin;
        const T inter_area = inter_w * inter_h;
        const T bbox1_area = BBoxArea<T>(box1, normalized);
        const T bbox2_area = BBoxArea<T>(box2, normalized);
        return inter_area / (bbox1_area + bbox2_area - inter_area);
    }
}

template <typename T>
static inline void NMSFast(
                           const T *bbox_data,
                           std::vector<int> bbox_dim,
                           const T *score_data,
                           const T score_threshold, const T nms_threshold,
                           const T eta, const int top_k,
                           std::vector<int>* selected_indices) {
    // The total boxes for each instance.
    int num_boxes = bbox_dim[0];
    // 4: [xmin ymin xmax ymax]
    int box_size = bbox_dim[1];
    
    std::vector<T> scores_data(num_boxes);
    std::copy_n(score_data, num_boxes, scores_data.begin());
    std::vector<std::pair<T, int>> sorted_indices;
    GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);
    
    selected_indices->clear();
    T adaptive_threshold = nms_threshold;
    
    while (sorted_indices.size() != 0) {
        const int idx = sorted_indices.front().second;
        bool keep = true;
        for (size_t k = 0; k < selected_indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*selected_indices)[k];
                T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                              bbox_data + kept_idx * box_size, true);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            selected_indices->push_back(idx);
        }
        sorted_indices.erase(sorted_indices.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

template <typename T>
void MultiClassNMS(const T *boxes_data,
                   const std::vector<int> &box_dim,
                   const T *scores_data,
                   const std::vector<int> &score_dim,
                   std::map<int, std::vector<int>>* indices, int* num_nmsed_out,
                   const int& background_label, const int& nms_top_k,
                   const int& keep_top_k, const T& nms_threshold,
                   const T& nms_eta, const T& score_threshold) {
    
    int64_t class_num = score_dim[0];
    int64_t predict_dim = score_dim[1];
    int num_det = 0;
    for (int c = 0; c < class_num; ++c) {
        if (c == background_label) continue;
        const T *score_data = scores_data + c * predict_dim;
        
        /// [c] is key
        NMSFast<T>(boxes_data, box_dim, score_data, score_threshold, nms_threshold, nms_eta,
                   nms_top_k, &((*indices)[c]));
        num_det += (*indices)[c].size();
    }
    
    *num_nmsed_out = num_det;
    if (keep_top_k > -1 && num_det > keep_top_k) {
        std::vector<std::pair<T, std::pair<int, int>>> score_index_pairs;
        for (const auto& it : *indices) {
            int label = it.first;
            const T* sdata = scores_data + label * predict_dim;
            const std::vector<int>& label_indices = it.second;
            for (size_t j = 0; j < label_indices.size(); ++j) {
                int idx = label_indices[j];
                // PADDLE_ENFORCE_LT(idx, predict_dim);
                score_index_pairs.push_back(std::make_pair(sdata[idx], std::make_pair(label, idx)));
            }
        }
        // Keep top k results per image.
        std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                         SortScorePairDescend<std::pair<int, int>>);
        score_index_pairs.resize(keep_top_k);
        
        // Store the new indices.
        std::map<int, std::vector<int>> new_indices;
        for (size_t j = 0; j < score_index_pairs.size(); ++j) {
            int label = score_index_pairs[j].second.first;
            int idx = score_index_pairs[j].second.second;
            new_indices[label].push_back(idx);
        }
        new_indices.swap(*indices);
        *num_nmsed_out = keep_top_k;
    }
}

template <typename T>
void MultiClassOutput(const T *scores_data,
                      const std::vector<int> &score_dim,
                      const T *bboxes_data,
                      T *outputs_data,
                      const std::map<int, std::vector<int>>& selected_indices) {
    int predict_dim = score_dim[1];
    int count = 0;
    for (const auto& it : selected_indices) {
        /// one batch
        int label = it.first;
        const T* sdata = scores_data + label * predict_dim;
        const std::vector<int>& indices = it.second;
        for (size_t j = 0; j < indices.size(); ++j) {
            int idx = indices[j];
            const T* bdata = bboxes_data + idx * kBBoxSize;
            outputs_data[count * kOutputDim] = label;           // label
            outputs_data[count * kOutputDim + 1] = sdata[idx];  // score
            // xmin, ymin, xmax, ymax
            std::memcpy(outputs_data + count * kOutputDim + 2, bdata, 4 * sizeof(T));
            count++;
        }
    }
}

void MultiClassNMSCompute(NMSParam *param) {
    assert(param->score_dim[0] == 1);
    assert(param->box_dim[0] == 1);
    assert (param->score_dim.size() == 3);
    assert(param->box_dim.size() == 3);
    
    float* outputs;
    auto background_label = param->background_label;
    auto nms_top_k = param->nmsTopK;
    auto keep_top_k = param->keepTopK;
    auto nms_threshold = param->nmsThreshold;
    auto nms_eta = param->nmsEta;
    auto score_threshold = param->scoreThredshold;
    
    std::vector<int> score_dim_one_batch = {param->score_dim[1], param->score_dim[2]};
    std::vector<int> box_dim_one_batch = {param->box_dim[1], param->box_dim[2]};
    
    std::vector<int> batch_starts = {0};
    
    std::map<int, std::vector<int>> indices;
    int num_nmsed_out = 0;
    
    MultiClassNMS<float>(param->box_data, box_dim_one_batch, param->score_data, score_dim_one_batch, &indices, &num_nmsed_out,
                         background_label, nms_top_k, keep_top_k, nms_threshold,
                         nms_eta, score_threshold);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    
    int output_size = 0;
    int num_kept = batch_starts.back();
    if (num_kept == 0) {
        outputs = new float[1];
        outputs[0] = -1;
        output_size = 1;
    } else {
        outputs = new float[num_kept * kOutputDim];
        int64_t s = batch_starts[0];
        int64_t e = batch_starts[1];
        if (e > s) {
            MultiClassOutput<float>(param->score_data, score_dim_one_batch, param->box_data, outputs, indices);
        }
        output_size = num_kept * kOutputDim;
    }
    param->output = outputs;
    param->output_size = output_size;
}

@implementation CPUResult
@end

@implementation NMSCompute

-(CPUResult *)computeWithScore:(float *)score andBBoxs:(float *)bbox {
    NMSParam param;
    param.box_data = bbox;
    param.score_data = score;
    param.background_label = self.background_label;
    param.scoreThredshold = self.scoreThredshold;
    param.nmsTopK = self.nmsTopK;
    param.keepTopK = self.keepTopK;
    param.nmsEta = self.nmsEta;
    param.nmsThreshold = self.nmsThreshold;
    std::vector<int> score_dim;
    for (int i = 0; i < self.scoreDim.count; ++i) {
        score_dim.push_back(self.scoreDim[i].intValue);
    }
    param.score_dim = score_dim;
    
    std::vector<int> box_dim;
    for (int i = 0; i < self.bboxDim.count; ++i) {
        box_dim.push_back(self.bboxDim[i].intValue);
    }
    param.box_dim = box_dim;
    MultiClassNMSCompute(&param);
    CPUResult *cr = [[CPUResult alloc] init];
    cr.output = param.output;
    cr.outputSize = param.output_size;
    return cr;
}

@end


