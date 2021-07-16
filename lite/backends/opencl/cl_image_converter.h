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

#pragma once

#include "lite/api/paddle_place.h"
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

class CLImageConverterBase {
 public:
  virtual ~CLImageConverterBase() {}

  virtual void NCHWToImage(float *nchw,
                           void *image,
                           const DDim &tensor_dim) = 0;

  virtual void ImageToNCHW(void *image,
                           float *nchw,
                           const DDim &image_dim,
                           const DDim &tensor_dim) = 0;
  virtual DDim InitImageDimInfoWith(const DDim &tensor_dim) = 0;

  bool fp16_support_{paddle::lite::CLRuntime::Global()->get_precision() ==
                     lite_api::CL_PRECISION_FP16};
};

class CLImageConverterDefault : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *nchw, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

class CLImageConverterFolder : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;

  /*
   *  width of original tensor
   * */
  inline size_t WidthOfOneBlock() const { return width_of_one_block_; }

  /*
   *  height of original tensor
   * */
  inline size_t HeightOfOneBlock() const { return height_of_one_block_; }

  int GetCBlock() const { return c_block_; }

 private:
  int c_block_;
  int width_of_one_block_;
  int height_of_one_block_;
};

class CLImageConverterNormal : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;

  /*
   *  width of original tensor
   * */
  inline size_t WidthOfOneBlock() const { return width_of_one_block_; }

  /*
   *  height of original tensor
   * */
  inline size_t HeightOfOneBlock() const { return height_of_one_block_; }

  int GetCBlock() const { return c_block_; }

 private:
  int c_block_;
  int width_of_one_block_;
  int height_of_one_block_;
};

class CLImageConverterNWBlock : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};
class CLImageConverterDWBlock : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

class CLImageConverterWinoTransWeight : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

class CLImageConverterNBlock : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

class CLImageConverterN2Block : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

class CLImageConverterDWFilter : public CLImageConverterBase {
 public:
  DDim InitImageDimInfoWith(const DDim &tensor_dim) override;
  void NCHWToImage(float *tensor, void *image, const DDim &tensor_dim) override;
  void ImageToNCHW(void *image,
                   float *tensor,
                   const DDim &image_dim,
                   const DDim &tensor_dim) override;
};

}  // namespace lite
}  // namespace paddle
