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

#include <gtest/gtest.h>
#include <fstream>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

static int randint(int beg, int end) {
  static unsigned int seed = 0;
  int rd = rand_r(&seed);
  int range = end - beg + 1;
  rd = rd % range;
  return rd + beg;
}
static constexpr int kROISize = 4;

template <class T>
void PreCalcForBilinearInterpolate(const int height,
                                   const int width,
                                   const int pooled_height,
                                   const int pooled_width,
                                   const int iy_upper,
                                   const int ix_upper,
                                   T roi_ymin,
                                   T roi_xmin,
                                   T bin_size_h,
                                   T bin_size_w,
                                   int roi_bin_grid_h,
                                   int roi_bin_grid_w,
                                   Tensor* pre_pos,
                                   Tensor* pre_w) {
  int pre_calc_index = 0;
  int* pre_pos_data = pre_pos->mutable_data<int>();
  T* pre_w_data = pre_w->mutable_data<T>();
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        // calculate y of sample points
        T y = roi_ymin + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h);
        // calculate x of samle points
        for (int ix = 0; ix < ix_upper; ix++) {
          T x = roi_xmin + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);
          // deal with elements out of map
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            for (int i = 0; i < kROISize; ++i) {
              pre_pos_data[i + pre_calc_index * kROISize] = 0;
              pre_w_data[i + pre_calc_index * kROISize] = 0;
            }
            pre_calc_index += 1;
            continue;
          }
          y = y <= 0 ? 0 : y;
          x = x <= 0 ? 0 : x;

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;
          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = static_cast<T>(y_low);
          } else {
            y_high = y_low + 1;
          }
          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = static_cast<T>(x_low);
          } else {
            x_high = x_low + 1;
          }
          T ly = y - y_low, lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          pre_pos_data[pre_calc_index * kROISize] = y_low * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 1] = y_low * width + x_high;
          pre_pos_data[pre_calc_index * kROISize + 2] = y_high * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 3] = y_high * width + x_high;
          pre_w_data[pre_calc_index * kROISize] = hy * hx;
          pre_w_data[pre_calc_index * kROISize + 1] = hy * lx;
          pre_w_data[pre_calc_index * kROISize + 2] = ly * hx;
          pre_w_data[pre_calc_index * kROISize + 3] = ly * lx;
          pre_calc_index += 1;
        }
      }
    }
  }
}

class RoiAlignComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string rois_ = "ROIs";
  std::string roislod_ = "RoisLod";
  std::string out_ = "Out";
  float spatial_scale_ = 0.5;
  int pooled_height_ = 2;
  int pooled_width_ = 2;
  int sampling_ratio_ = -1;
  bool test_fluid_v18_api_ = false;

 public:
  RoiAlignComputeTester(const Place& place,
                        const std::string& alias,
                        bool test_fluid_v18_api)
      : TestCase(place, alias), test_fluid_v18_api_(test_fluid_v18_api) {}

  DDim stride(const DDim& ddim) {
    DDim strides = ddim;
    strides[ddim.size() - 1] = 1;
    for (int i = ddim.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * ddim[i + 1];
    }
    return strides;
  }
  template <class T>
  void Compute(Scope* scope) {
    auto* in = scope->FindTensor(x_);
    auto* rois = scope->FindTensor(rois_);
    auto pooled_height = pooled_height_;
    auto pooled_width = pooled_width_;
    auto spatial_scale = spatial_scale_;
    auto sampling_ratio = sampling_ratio_;

    auto in_dims = in->dims();
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    DDim out_dims(std::vector<int64_t>(
        {rois_num, channels, pooled_height_, pooled_width_}));

    auto* out = scope->NewTensor(out_);
    out->Resize(out_dims);

    auto in_stride = stride(in_dims);
    auto roi_stride = stride(rois->dims());
    auto out_stride = stride(out->dims());

    const T* input_data = in->data<T>();
    lite::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>();
    int rois_batch_size;
    if (test_fluid_v18_api_) {
      auto* rois_lod_t = scope->FindTensor(roislod_);
      rois_batch_size = rois_lod_t->numel();
      auto* rois_lod = rois_lod_t->data<int64_t>();
      for (int n = 0; n < rois_batch_size - 1; ++n) {
        for (int i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    } else {
      auto lod = rois->lod();
      auto rois_lod = lod.back();
      int rois_batch_size = rois_lod.size() - 1;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    }
    T* output_data = out->mutable_data<T>();
    const T* rois_data = rois->data<T>();
    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_id = roi_batch_id_data[n];
      T roi_xmin = rois_data[0] * spatial_scale;
      T roi_ymin = rois_data[1] * spatial_scale;
      T roi_xmax = rois_data[2] * spatial_scale;
      T roi_ymax = rois_data[3] * spatial_scale;

      T roi_width = std::max(roi_xmax - roi_xmin, static_cast<T>(1.));
      T roi_height = std::max(roi_ymax - roi_ymin, static_cast<T>(1.));
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      const T* batch_data = input_data + roi_batch_id * in_stride[0];

      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);
      const T count = roi_bin_grid_h * roi_bin_grid_w;
      Tensor pre_pos;
      Tensor pre_w;
      int pre_size = count * out_stride[1];
      pre_pos.Resize({pre_size, kROISize});
      pre_w.Resize({pre_size, kROISize});

      PreCalcForBilinearInterpolate(height,
                                    width,
                                    pooled_height,
                                    pooled_width,
                                    roi_bin_grid_h,
                                    roi_bin_grid_w,
                                    roi_ymin,
                                    roi_xmin,
                                    bin_size_h,
                                    bin_size_w,
                                    roi_bin_grid_h,
                                    roi_bin_grid_w,
                                    &pre_pos,
                                    &pre_w);
      const int* pre_pos_data = pre_pos.data<int>();
      const T* pre_w_data = pre_w.data<T>();
      for (int c = 0; c < channels; c++) {
        int pre_calc_index = 0;
        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            const int pool_index = ph * pooled_width + pw;
            T output_val = 0;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                for (int i = 0; i < kROISize; i++) {
                  int pos = pre_pos_data[pre_calc_index * kROISize + i];
                  T w = pre_w_data[pre_calc_index * kROISize + i];
                  output_val += w * batch_data[pos];
                }
                pre_calc_index += 1;
              }
            }
            output_val /= count;
            output_data[pool_index] = output_val;
          }
        }
        batch_data += in_stride[1];
        output_data += out_stride[1];
      }
      rois_data += roi_stride[0];
    }
  }

  void RunBaseline(Scope* scope) override { Compute<float>(scope); }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("roi_align");

    op_desc->SetInput("X", {x_});
    op_desc->SetInput("ROIs", {rois_});
    if (test_fluid_v18_api_) {
      op_desc->SetInput("RoisLod", {roislod_});
    }

    op_desc->SetAttr("spatial_scale", spatial_scale_);
    op_desc->SetAttr("pooled_height", pooled_height_);
    op_desc->SetAttr("pooled_width", pooled_width_);
    op_desc->SetAttr("sampling_ratio", sampling_ratio_);

    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    int batch_size = 3;
    int channels = 3;
    int height = 8;
    int width = 6;

    // setup x
    {
      DDim dims(std::vector<int64_t>({batch_size, channels, height, width}));
      std::vector<float> datas;
      datas.resize(dims.production());
      std::generate(datas.begin(), datas.end(), std::rand);
      SetCommonTensor(x_, dims, datas.data());
    }

    {
      auto rois = std::vector<float>();
      DDim rois_dims(
          std::vector<int64_t>({batch_size * (batch_size + 1) / 2, 4}));
      auto rois_lod0 = std::vector<uint64_t>(1, 0);
      for (int bno = 0; bno < batch_size; ++bno) {
        uint64_t new_end = *rois_lod0.rbegin() + bno + 1;
        rois_lod0.push_back(new_end);
        for (int i = 0; i < (bno + 1); ++i) {
          float x1 = randint(0, width / spatial_scale_ - pooled_width_);
          float y1 = randint(0, height / spatial_scale_ - pooled_height_);

          float x2 = randint(x1 + pooled_width_, width / spatial_scale_);
          float y2 = randint(y1 + pooled_height_, height / spatial_scale_);

          auto roi = std::vector<float>{x1, y1, x2, y2};
          rois.insert(rois.end(), roi.begin(), roi.end());
        }
      }

      if (test_fluid_v18_api_) {
        int64_t lod_size = rois_lod0.size();
        DDim lod_dims(std::vector<int64_t>({lod_size}));
        SetCommonTensor(rois_, rois_dims, rois.data());
        SetCommonTensor(roislod_, lod_dims, rois_lod0.data());
      } else {
        LoD lod;
        lod.push_back(rois_lod0);
        SetCommonTensor(rois_, rois_dims, rois.data(), lod);
      }
    }
  }
};

TEST(RoiAlign, precision) {
  // The unit test for roi_align needs the params,
  // which is obtained by runing model by paddle.
  LOG(INFO) << "test roi align op";
#ifdef LITE_WITH_ARM
  {
    Place place(TARGET(kARM));
    std::unique_ptr<arena::TestCase> tester(
        new RoiAlignComputeTester(place, "def", false));
    arena::Arena arena(std::move(tester), place, 2e-4);
    EXPECT_TRUE(arena.TestPrecision());
  }
  {
    Place place(TARGET(kARM));
    std::unique_ptr<arena::TestCase> tester(
        new RoiAlignComputeTester(place, "def", true));
    arena::Arena arena(std::move(tester), place, 2e-4);
    EXPECT_TRUE(arena.TestPrecision());
  }
#endif
}

}  // namespace lite
}  // namespace paddle
