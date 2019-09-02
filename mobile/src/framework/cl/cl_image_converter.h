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

#include "framework/cl/cl_half.h"
#include "framework/ddim.h"

namespace paddle_mobile {
namespace framework {

class CLImageConverterBase {
 public:
  virtual void NCHWToImage(float *nchw, half_t *image,
                           const DDim &tensor_dim) = 0;

  virtual void ImageToNCHW(half_t *image, float *nchw, const DDim &image_dim,
                           const DDim &tensor_dim) = 0;
  virtual const DDim &InitImageDimInfoWith(const DDim &tensor_dim) = 0;
};

class CLImageConverterDefault : public CLImageConverterBase {
 public:
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *nchw, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);
};

class CLImageConverterFolder : public CLImageConverterBase {
 public:
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);

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
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);

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
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);
};
class CLImageConverterDWBlock : public CLImageConverterBase {
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);
};

class CLImageConverterWinoTransWeight : public CLImageConverterBase {
 public:
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);
};

class CLImageConverterConv2dTransposeTransWeight : public CLImageConverterBase {
 public:
  const DDim &InitImageDimInfoWith(const DDim &tensor_dim);
  void NCHWToImage(float *tensor, half_t *image, const DDim &tensor_dim);
  void ImageToNCHW(half_t *image, float *tensor, const DDim &image_dim,
                   const DDim &tensor_dim);
};

}  // namespace framework
}  // namespace paddle_mobile
