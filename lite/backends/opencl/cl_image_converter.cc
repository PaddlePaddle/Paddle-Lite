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

#include "lite/backends/opencl/cl_image_converter.h"
#include <vector>
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
DDim CLImageConverterDefault::InitImageDimInfoWith(const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterDefault::NCHWToImage(float *nchw,
                                          void *image,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t width = in_image_dim[0];
  size_t w_block = width / W;

  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  float *p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * width * H + h * width + (c / 4) * W + w) * 4 +
            // (c % 4);
            fp16_support_ ? image_fp16[i2] = Float2Half(*p) : image_fp32[i2] =
                                                                  *p;
            i2 += 4;
            p++;
          } else {
            fp16_support_ ? image_fp16[i2] = Float2Half(0.f) : image_fp32[i2] =
                                                                   0.f;
            i2 += 4;
          }
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

void CLImageConverterDefault::ImageToNCHW(void *image,
                                          float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  size_t width = image_dim[0];
  float *p = tensor;
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          *p = fp16_support_ ? Half2Float(image_fp16[i2]) : image_fp32[i2];
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

DDim CLImageConverterFolder::InitImageDimInfoWith(const DDim &tensor_dim) {
  if (tensor_dim.size() <= 2) {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }
    size_t width = (tdim[1] + 3) / 4;
    size_t height = tdim[0];

    width_of_one_block_ = width;
    height_of_one_block_ = height;
    c_block_ = 1;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));

  } else {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }
    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];
    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;

    width_of_one_block_ = W;
    height_of_one_block_ = H;
    c_block_ = width / W;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));
  }
}

void CLImageConverterFolder::NCHWToImage(float *tensor,
                                         void *image,
                                         const DDim &tensor_dim) {
  CHECK(tensor_dim.size() <= 4 && tensor_dim.size() > 0)
      << " Tensor dim is not support!";

  if (tensor_dim.size() > 2) {
    CLImageConverterDefault default_converter;
    default_converter.NCHWToImage(tensor, image, tensor_dim);

  } else {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }

    DDim image_dim = InitImageDimInfoWith(tensor_dim);
    size_t width = image_dim[0];
    float *image_fp32 = static_cast<float *>(image);
    half_t *image_fp16 = static_cast<half_t *>(image);

    for (size_t h = 0; h < tdim[0]; h++) {
      for (size_t w = 0; w < width * 4; w++) {
        if (w < tdim[1]) {
          if (fp16_support_) {
            image_fp16[(h * width + w / 4) * 4 + (w % 4)] =
                Float2Half(tensor[h * tdim[1] + w]);
          } else {
            image_fp32[(h * width + w / 4) * 4 + (w % 4)] =
                tensor[h * tdim[1] + w];
          }
        } else {
          if (fp16_support_) {
            image_fp16[(h * width + w / 4) * 4 + (w % 4)] = Float2Half(0.f);
          } else {
            image_fp32[(h * width + w / 4) * 4 + (w % 4)] = 0.f;
          }
        }
      }
    }
  }
}

void CLImageConverterFolder::ImageToNCHW(void *image,
                                         float *tensor,
                                         const DDim &image_dim,
                                         const DDim &tensor_dim) {
  if (tensor_dim.size() > 2) {
    CLImageConverterDefault default_converter;
    default_converter.ImageToNCHW(image, tensor, image_dim, tensor_dim);

  } else {
    size_t width = image_dim[0];
    size_t H = 1, W = 1;

    if (tensor_dim.size() == 2) {
      H = tensor_dim[0];
      W = tensor_dim[1];
    } else if (tensor_dim.size() == 1) {
      W = tensor_dim[0];
    }

    float *p = tensor;
    float *image_fp32 = static_cast<float *>(image);
    half_t *image_fp16 = static_cast<half_t *>(image);

    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        p[h * W + w] =
            fp16_support_
                ? Half2Float(image_fp16[(h * width + w / 4) * 4 + (w % 4)])
                : image_fp32[(h * width + w / 4) * 4 + (w % 4)];
      }
    }
  }
}

DDim CLImageConverterNWBlock::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = W * ((N + 3) / 4);
  size_t height = C * H;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterNWBlock::NCHWToImage(float *tensor,
                                          void *image,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  auto image_dim = InitImageDimInfoWith(tensor_dim);
  float *p = tensor;
  size_t N = tensor_dim[0];
  size_t C = tensor_dim[1];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];
  size_t height = image_dim[1];
  size_t block = image_dim[0] / tensor_dim[3];
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  for (size_t n = 0; n < block * 4; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          size_t index = 4 * c * (width * H) + 4 * h * width + 4 * W * (n / 4) +
                         w * 4 + n % 4;
          if (n < N) {
            if (fp16_support_) {
              image_fp16[index] = Float2Half(*p);
            } else {
              image_fp32[index] = *p;
            }
            p++;
          } else {
            if (fp16_support_) {
              image_fp16[index] = Float2Half(0.f);
            } else {
              image_fp32[index] = 0.f;
            }
          }
          if (index >= (width * height * 4)) {
            LOG(INFO) << " index out of range ";
          }
        }
      }
    }
  }
  VLOG(3) << " init done";
}

void CLImageConverterNWBlock::ImageToNCHW(void *image,
                                          float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  float *p = tensor;
  size_t N = tensor_dim[0];
  size_t C = tensor_dim[1];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];
  size_t height = image_dim[1];
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          size_t index = 4 * c * (width * H) + 4 * h * width + 4 * W * (n / 4) +
                         w * 4 + n % 4;
          *p =
              fp16_support_ ? Half2Float(image_fp16[index]) : image_fp32[index];
          p++;
          if (index >= (width * height * 4)) {
            LOG(INFO) << " index out of range ";
          }
        }
      }
    }
  }
  VLOG(3) << " init done";
}

DDim CLImageConverterDWBlock::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = W * ((N + 3) / 4);
  size_t height = C * H;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterDWBlock::NCHWToImage(float *tensor,
                                          void *image,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[1];
  C = new_dims[0];
  H = new_dims[2];
  W = new_dims[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t width = in_image_dim[0];
  size_t w_block = width / W;

  float *p = tensor;
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * width * H + h * width + (c / 4) * W + w) * 4 +
            // (c % 4);
            if (fp16_support_) {
              image_fp16[i2] = Float2Half(*p);
            } else {
              image_fp32[i2] = *p;
            }
            i2 += 4;
            p++;
          } else {
            if (fp16_support_) {
              image_fp16[i2] = Float2Half(0.f);
            } else {
              image_fp32[i2] = 0.f;
            }
            i2 += 4;
          }
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

void CLImageConverterDWBlock::ImageToNCHW(void *image,
                                          float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  float *p = tensor;
  size_t N = tensor_dim[1];
  size_t C = tensor_dim[0];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          *p = fp16_support_ ? Half2Float(image_fp16[i2]) : image_fp32[i2];
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

DDim CLImageConverterNormal::InitImageDimInfoWith(const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;

  width_of_one_block_ = W;
  height_of_one_block_ = H;
  c_block_ = width / W;

  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterNormal::NCHWToImage(float *tensor,
                                         void *image,
                                         const DDim &tensor_dim) {
  CHECK(tensor_dim.size() <= 4 && tensor_dim.size() > 0)
      << " Tensor dim is not support!";

  CLImageConverterDefault default_converter;
  default_converter.NCHWToImage(tensor, image, tensor_dim);
}

void CLImageConverterNormal::ImageToNCHW(void *image,
                                         float *tensor,
                                         const DDim &image_dim,
                                         const DDim &tensor_dim) {
  CLImageConverterDefault default_converter;
  default_converter.ImageToNCHW(image, tensor, image_dim, tensor_dim);
}

DDim CLImageConverterWinoTransWeight::InitImageDimInfoWith(
    const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C;
  N = tensor_dim[0];
  C = tensor_dim[1];
  size_t width = ((C + 3) / 4) * 4;
  size_t height =
      ((N + 3) / 4) * 16;  // N * (wino_blk_size + 2) * (wino_blk_size + 2)
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}
static void matmul(float *C, float *A, float *B, int h, int k, int w) {
  float *a = A;
  float *b = B;
  float *c = C;

  const int aw = k;
  const int bw = w;
  const int cw = w;
  for (int y = 0; y < h; ++y) {
    const auto aLine = a + y * aw;
    auto cLine = c + y * cw;
    for (int x = 0; x < w; ++x) {
      auto bColumn = b + x;
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum += aLine[i] * bColumn[i * bw];
      }
      cLine[x] = sum;
    }
  }
}
void CLImageConverterWinoTransWeight::NCHWToImage(float *tensor,
                                                  void *image,
                                                  const DDim &tensor_dim) {
  std::vector<float> G = {
      1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

  std::vector<float> GT = {
      1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  int co = tensor_dim[0];
  int ci = tensor_dim[1];
  int kernelCount = tensor_dim[2];
  int unitCi = 4;
  int unitCo = 4;
  int alpha = 4;
  int num_count = 16 * ((co + 3) / 4) * ((ci + 3) / 4) * 4 * 4;
  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);
  // auto weight_dest_data = static_cast<half_t *>(image);
  if (fp16_support_) {
    memset(image_fp16, 0, num_count * sizeof(half_t));
  } else {
    memset(image_fp32, 0, num_count * sizeof(float));
  }

  std::vector<float> M(12);
  std::vector<float> K_Transform(16);
  auto weightPtr = tensor;

  int oz_index, alpha_index;
  alpha_index = 0;
  oz_index = 1;

  for (int oz = 0; oz < co; ++oz) {
    auto srcOz = weightPtr + oz * ci * kernelCount * kernelCount;

    int ozC4 = oz / unitCo;
    int mx = oz % unitCo;
    auto dstOz_fp16 = image_fp16 + ((ci + 3) / 4) * 4 * 4 * ozC4 + mx;
    auto dstOz_fp32 = image_fp32 + ((ci + 3) / 4) * 4 * 4 * ozC4 + mx;
    for (int sz = 0; sz < ci; ++sz) {
      int szC4 = sz / unitCi;
      int my = sz % unitCi;
      auto srcSz = srcOz + kernelCount * kernelCount * sz;

      matmul(M.data(), G.data(), srcSz, 4, 3, 3);

      matmul(K_Transform.data(), M.data(), GT.data(), 4, 3, 4);

      auto dstSz_fp16 = dstOz_fp16 + szC4 * 16 + unitCo * my;
      auto dstSz_fp32 = dstOz_fp32 + szC4 * 16 + unitCo * my;
      for (int i = 0; i < 16; ++i) {
        if (fp16_support_) {
          *(dstSz_fp16 + i * ((co + 3) / 4) * ((ci + 3) / 4) * 4 * 4) =
              Float2Half(K_Transform.data()[i]);
        } else {
          *(dstSz_fp32 + i * ((co + 3) / 4) * ((ci + 3) / 4) * 4 * 4) =
              K_Transform.data()[i];
        }
      }
    }
  }
}

void CLImageConverterWinoTransWeight::ImageToNCHW(void *image,
                                                  float *tensor,
                                                  const DDim &image_dim,
                                                  const DDim &tensor_dim) {}

DDim CLImageConverterNBlock::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = ((C + 3) / 4) * 4;
  size_t height = ((N + 3) / 4) * H * W;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

DDim CLImageConverterNBlockGroup::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = ((C + 3) / 4) * 4;
  size_t height = ((N / groups + 3) / 4 * groups) * H * W;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterNBlock::NCHWToImage(float *nchw,
                                         void *image,
                                         const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t height = in_image_dim[1];
  size_t n_block = height / (W * H);
  size_t c_block4 = in_image_dim[0];

  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  float *p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < n_block * 4; n++) {
    for (size_t c = 0; c < c_block4; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t img_idx =
              (((n / 4) * W * H + h * W + w) * c_block4 + c) * 4 + n % 4;
          if (n < N && c < C) {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(*p)
                          : image_fp32[img_idx] = *p;
            p++;
          } else {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(0.f)
                          : image_fp32[img_idx] = 0.f;
          }
        }
      }
    }
  }
}

void CLImageConverterNBlockGroup::ImageToNCHW(void *image,
                                              float *tensor,
                                              const DDim &image_dim,
                                              const DDim &tensor_dim) {}

void CLImageConverterNBlockGroup::NCHWToImage(float *nchw,
                                              void *image,
                                              const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t height = in_image_dim[1];
  size_t n_block = height / (W * H);
  size_t c_block4 = ((in_image_dim[0] + 3) / 4) * 4;

  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  float *p = nchw;
  size_t i0 = 0;
  int i = 0;
  for (size_t n = 0; n < n_block * 4; n++) {
    for (size_t c = 0; c < c_block4; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t img_idx =
              (((n / 4) * W * H + h * W + w) * c_block4 + c) * 4 + n % 4;
          size_t remain = n % ((N / groups + 3) / 4 * 4);
          if (remain < (N / groups) && c < C) {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(*p)
                          : image_fp32[img_idx] = *p;
            p++;
          } else {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(0.f)
                          : image_fp32[img_idx] = 0.f;
          }
        }
      }
    }
  }
}

void CLImageConverterNBlock::ImageToNCHW(void *image,
                                         float *tensor,
                                         const DDim &image_dim,
                                         const DDim &tensor_dim) {}

DDim CLImageConverterN2Block::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = (C + 3) / 4 * 2 * 4;
  size_t height = ((N + 7) / 8) * H * W;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterN2Block::NCHWToImage(float *nchw,
                                          void *image,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t height = in_image_dim[1];
  size_t n_block = height / (W * H);
  size_t c_block = (C + 3) / 4;

  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  float *p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < n_block * 8; n++) {
    for (size_t c = 0; c < c_block * 4; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t img_idx = ((n / 8) * W * H + h * W + w) * c_block * 4 * 8 +
                           (c / 4) * 32 + ((n % 8) / 4) * 16 + (c % 4) * 4 +
                           (n % 8) % 4;
          if (n < N && c < C) {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(*p)
                          : image_fp32[img_idx] = *p;
            p++;
          } else {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(0.f)
                          : image_fp32[img_idx] = 0.f;
          }
        }
      }
    }
  }
}

void CLImageConverterN2Block::ImageToNCHW(void *image,
                                          float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {}

DDim CLImageConverterDWFilter::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = H * W;
  size_t height = ((N + 3) / 4) * C;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterDWFilter::NCHWToImage(float *nchw,
                                           void *image,
                                           const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);
  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t height = in_image_dim[1];
  size_t n_block = height / C;

  float *image_fp32 = static_cast<float *>(image);
  half_t *image_fp16 = static_cast<half_t *>(image);

  float *p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < n_block * 4; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t img_idx = (((n / 4) * W * H + h * W + w) * C + c) * 4 + n % 4;
          if (n < N) {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(*p)
                          : image_fp32[img_idx] = *p;
            p++;
          } else {
            fp16_support_ ? image_fp16[img_idx] = Float2Half(0.f)
                          : image_fp32[img_idx] = 0.f;
          }
        }
      }
    }
  }
}

void CLImageConverterDWFilter::ImageToNCHW(void *image,
                                           float *tensor,
                                           const DDim &image_dim,
                                           const DDim &tensor_dim) {}

}  // namespace lite
}  // namespace paddle
