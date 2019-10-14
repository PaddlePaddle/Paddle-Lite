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

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++ features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace paddle_mobile {

#ifdef PADDLE_MOBILE_FPGA

namespace fpga {
int open_device();
int close_device();
void* fpga_malloc(size_t size);
void fpga_free(void* ptr);

//  Usage:
//  auto version = fpga::paddle_mobile_version();
//  std::cout << "0X0" << std::hex << version << std::endl;
uint32_t paddle_mobile_version();
}  // namespace fpga
#endif

enum PaddleDType {
  FLOAT32,
  FLOAT16,
  INT64,
  INT8,
  UINT8,
};

enum LayoutType {
  LAYOUT_CHW = 1,
  LAYOUT_HWC = 0,
};

class PaddleBuf {
 public:
  PaddleBuf() = default;
  PaddleBuf(PaddleBuf&& other);
  // Copy only available when memory is managed externally.
  explicit PaddleBuf(const PaddleBuf&);
  PaddleBuf& operator=(const PaddleBuf&);
  // Do not own the memory.
  PaddleBuf(void* data, size_t length)
      : data_(data), length_(length), memory_owned_{false} {}
  // Own memory.
  explicit PaddleBuf(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  // Resize to `length` bytes.
  void Resize(size_t length);
  // Reset to external memory.
  void Reset(void* data, size_t length);
  bool empty() const { return length_ == 0; }
  void* data() const { return data_; }
  size_t length() const { return length_; }

  ~PaddleBuf() { Free(); }

 private:
  void Free();
  void* data_{nullptr};  // pointer to the data memory.
  size_t length_{0};     // number of memory bytes.
  bool memory_owned_{true};
};

typedef enum {
  paddle_void = 0,
  paddle_float,
  paddle_int,
  paddle_uint16_t,
  paddle_double,
  paddle_int64_t,
  paddle_size_t,
  paddle_int16_t,
  paddle_int8_t,
  paddle_uint8_t,
  paddle_bool,
  paddle_string,
  paddle_floats = 100,
  paddle_ints,
  paddle_int64_ts,
  paddle_size_ts,
  paddle_bools,
  paddle_strings,
  paddle_const_float = 200,
  paddle_const_int,
  paddle_block = 300,
  paddle_tensor,
  paddle_lod_tensor,
  paddle_blocks,
  paddle_tensors,
  paddle_lod_tensors,
  paddle_p_block = 400,
  paddle_p_tensor,
  paddle_p_lod_tensor,
  paddle_p_blocks,
  paddle_p_tensors,
  paddle_p_lod_tensors,
  paddle_scopes = 500,
  paddle_selected_rows,
  paddle_dim0 = 600,
  paddle_dim1,
  paddle_dim2,
  paddle_dim3,
  paddle_dim4,
  paddle_dim5,
  paddle_dim6,
  paddle_dim7,
  paddle_dim8,
  paddle_dim9,
#ifdef PADDLE_MOBILE_CL
  paddle_cl_image,
#endif
} PaddlekTypeId_t;

struct PaddleTensor {
  PaddleTensor() = default;
  std::string name;  // variable name.
  std::vector<int> shape;
  std::vector<int> lod;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
  PaddlekTypeId_t dtypeid;
  LayoutType layout;
};

enum class PaddleEngineKind {
  kPaddleMobile,
  // TODO(Superjomn) support following engines latter.
  // kTensorRT,           // Use TensorRT for inference.
  // kAutoMixedAnakin,    // Automatically mix Fluid with Anakin.
  // kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
};

/*
 * A simple Inference API for Paddle. Currently this API can be used by
 * non-sequence scenerios.
 */
class PaddlePredictor {
 public:
  struct Config;
  PaddlePredictor(const PaddlePredictor&) = delete;
  PaddlePredictor& operator=(const PaddlePredictor&) = delete;

  // Predict an record.
  // The caller should be responsible for allocating and releasing the memory of
  // `inputs`. `inputs` should be available until Run returns. Caller should be
  // responsible for the output tensor's buffer, either allocated or passed from
  // outside.

  virtual bool Run(const std::vector<PaddleTensor>& inputs,
                   std::vector<PaddleTensor>* output_data,
                   int batch_size = -1) = 0;
  // Destroy the Predictor.
  virtual ~PaddlePredictor() = default;

  // The common configs for all the predictors.
  struct Config {
    std::string model_dir;  // path to the model directory.
    std::string prog_file;
    std::string param_file;
  };
#ifdef PADDLE_MOBILE_FPGA
  virtual void Predict_From_To(int start, int end) = 0;
  virtual void FeedPaddleTensors(const std::vector<PaddleTensor>& inputs) = 0;
  virtual void FetchPaddleTensors(std::vector<PaddleTensor>* outputs) = 0;
  virtual void FetchPaddleTensors(PaddleTensor* outputs, int id) = 0;
  virtual void GetPaddleTensor(const std::string& name,
                               PaddleTensor* output) = 0;
#endif

 protected:
  PaddlePredictor() = default;
};

struct PaddleModelMemoryPack {
  bool from_memory = false;
  size_t model_size = 0;
  uint8_t* model_buf = nullptr;
  size_t combined_params_size = 0;
  uint8_t* combined_params_buf = nullptr;
};

struct PaddleMobileConfig : public PaddlePredictor::Config {
  enum Precision { FP32 = 0 };
  enum Device { kCPU = 0, kFPGA = 1, kGPU_MALI = 2, kGPU_CL = 3 };
  enum PrePostType { NONE_PRE_POST = 0, UINT8_255 = 1 };

  enum Precision precision;
  enum Device device;
  enum PrePostType pre_post_type;

  int batch_size = 1;
  bool optimize = true;
  bool quantification = false;
  int quantification_fold = 1;
  bool lod_mode = false;
  int thread_num = 1;
  bool load_when_predict = false;
  std::string cl_path;
  struct PaddleModelMemoryPack memory_pack;
};

// A factory to help create different predictors.
template <typename ConfigT,
          PaddleEngineKind engine = PaddleEngineKind::kPaddleMobile>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

}  // namespace paddle_mobile
