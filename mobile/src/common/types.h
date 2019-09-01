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

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace paddle_mobile {
enum class Precision : int { FP32 = 0, FP16 = 1 };

typedef int16_t half;

template <Precision p>
struct PrecisionTrait {
  typedef void ptype;
};

template <>
struct PrecisionTrait<Precision::FP32> {
  typedef float ptype;
};
template <>
struct PrecisionTrait<Precision::FP16> {
  typedef half ptype;
};

//! device type
enum DeviceTypeEnum {
  kINVALID = -1,
  kCPU = 0,
  kFPGA = 1,
  kGPU_MALI = 2,
  kGPU_CL = 3
};

template <DeviceTypeEnum T>
struct DeviceType {};

typedef DeviceType<kCPU> CPU;
typedef DeviceType<kFPGA> FPGA;
typedef DeviceType<kGPU_CL> GPU_CL;

//! data type
enum DataType {
  PM_INVALID = -1,
  PM_HALF = 0,
  PM_FLOAT = 1,
  PM_DOUBLE = 2,
  PM_INT8 = 3,
  PM_INT16 = 4,
  PM_INT32 = 5,
  PM_INT64 = 6,
  PM_UINT8 = 7,
  PM_UINT16 = 8,
  PM_UINT32 = 9,
  PM_STRING = 10,
  PM_BOOL = 11,
  PM_SHAPE = 12,
  PM_TENSOR = 13
};
//!
enum PMStatus {
  PMSuccess = 0xFF,        /*!< No errors */
  PMNotInitialized = 0x01, /*!< Data not initialized. */
  PMInvalidValue = 0x02,   /*!< Incorrect variable value. */
  PMMemAllocFailed = 0x03, /*!< Memory allocation error. */
  PMUnKownError = 0x04,    /*!< Unknown error. */
  PMOutOfAuthority = 0x05, /*!< Try to modified data not your own*/
  PMOutOfMem = 0x06,       /*!< OOM error*/
  PMUnImplError = 0x07,    /*!< Unimplement error. */
  PMWrongDevice = 0x08,    /*!< un-correct device. */
  PMException = 0x09       /*!< throw exception. */
};

enum RoundType {
  ROUND_NEAREST_AWAY_ZERO = 0,
  ROUND_NEAREST_TOWARDS_ZERO = 1,
  ROUND_NEAREST_TO_EVEN = 2,
};

enum ActivationType {
  IDENTITY = 0,
  RELU = 1,
  RELU6 = 2,
  PRELU = 3,
  LEAKY_RELU = 4,
  TANH = 5,
  SIGMOID = 6,
  LOG = 7,
};

enum PoolingType {
  MAX = 0,
  AVG = 1,
  SUM = 2,
  FIRST = 3,
  LAST = 4,
};

enum PowerMode {
  PERFORMANCE_PRIORITY = 0,  // let threads run on big cores if
                             // thread_num <= big_cores_num,
                             // otherwise the power mode will be
                             // set to AUTO and all threads are
                             // scheduled by system
  EFFICIENCY_PRIORITY = 1,   // let threads run on little cores if
                             // thread_num <= little_cores_num,
                             // otherwise the power mode will be
                             // set to AUTO and all threads are
                             // scheduled by system
  PERFORMANCE_ONLY = 2,      // force threads run on big cores,
                             // and the remains are ignored if
                             // exceed the number big cores
  EFFICIENCY_ONLY = 3,       // force threads run on little cores,
                             // and the remains are ignored if
                             // exceed the number of little cores
  AUTO = 4,                  // scheduled by system
};

enum MemoryOptimizationLevel {
  NoMemoryOptimization = 0,
  MemoryOptimizationWithoutFeeds = 1,
  FullMemoryOptimization = 2,
};

struct PaddleMobileConfigInternal {
  bool load_when_predict = false;
  MemoryOptimizationLevel memory_optimization_level =
      MemoryOptimizationWithoutFeeds;
  std::string model_obfuscate_key = "";
};

enum ARMArch {
  APPLE = 0,
  A53 = 53,
  A55 = 55,
  A57 = 57,
  A72 = 72,
  A73 = 73,
  A75 = 75,
  A76 = 76,
  ARM_UNKOWN = -1
};

extern const char *G_OP_TYPE_CONV;
extern const char *G_OP_TYPE_BATCHNORM;
extern const char *G_OP_TYPE_INSTANCENORM;
extern const char *G_OP_TYPE_BOX_CODER;
extern const char *G_OP_TYPE_CONCAT;
extern const char *G_OP_TYPE_ELEMENTWISE_ADD;
extern const char *G_OP_TYPE_ELEMENTWISE_SUB;
extern const char *G_OP_TYPE_ELEMENTWISE_MUL;
extern const char *G_OP_TYPE_FUSION_CONV_ADD_RELU;
extern const char *G_OP_TYPE_FUSION_CONV_ADD_PRELU;
extern const char *G_OP_TYPE_FUSION_CONV_ADD_ADD_PRELU;
extern const char *G_OP_TYPE_FC;
extern const char *G_OP_TYPE_FUSION_CONV_ADD;
extern const char *G_OP_TYPE_FUSION_CONV_ADD_BN_RELU;
extern const char *G_OP_TYPE_FUSION_CONV_BN_ADD_RELU;
extern const char *G_OP_TYPE_FUSION_DWCONV_BN_RELU;
extern const char *G_OP_TYPE_FUSION_CONV_BN_RELU;
extern const char *G_OP_TYPE_FUSION_CONV_RELU;

extern const char *G_OP_TYPE_GRU;
extern const char *G_OP_TYPE_GRU_UNIT;
extern const char *G_OP_TYPE_CRF;
extern const char *G_OP_TYPE_BILINEAR_INTERP;
extern const char *G_OP_TYPE_NEAREST_INTERP;
extern const char *G_OP_TYPE_FLATTEN;
extern const char *G_OP_TYPE_FLATTEN2;
extern const char *G_OP_TYPE_SHAPE;
extern const char *G_OP_TYPE_LRN;
extern const char *G_OP_TYPE_MUL;
extern const char *G_OP_TYPE_MULTICLASS_NMS;
extern const char *G_OP_TYPE_NORM;
extern const char *G_OP_TYPE_POOL2D;
extern const char *G_OP_TYPE_PRIOR_BOX;
extern const char *G_OP_TYPE_RELU;
extern const char *G_OP_TYPE_RELU6;
extern const char *G_OP_TYPE_LEAKY_RELU;
extern const char *G_OP_TYPE_RESHAPE;
extern const char *G_OP_TYPE_SCALE;
extern const char *G_OP_TYPE_SIGMOID;
extern const char *G_OP_TYPE_SOFTMAX;
extern const char *G_OP_TYPE_TRANSPOSE;
extern const char *G_OP_TYPE_SPLIT;
extern const char *G_OP_TYPE_FEED;
extern const char *G_OP_TYPE_FETCH;
extern const char *G_OP_TYPE_DEPTHWISE_CONV;
extern const char *G_OP_TYPE_IM2SEQUENCE;
extern const char *G_OP_TYPE_DROPOUT;

extern const char *G_OP_TYPE_FUSION_CONV_ADD_BN;
extern const char *G_OP_TYPE_FUSION_POOL_BN;
extern const char *G_OP_TYPE_FUSION_ELEMENTWISE_ADD_RELU;
extern const char *G_OP_TYPE_FUSION_FC_RELU;
extern const char *G_OP_TYPE_REGION;
extern const char *G_OP_TYPE_FUSION_CONV_BN;
extern const char *G_OP_TYPE_CONV_TRANSPOSE;
extern const char *G_OP_TYPE_PRELU;
extern const char *G_OP_TYPE_SUM;
extern const char *G_OP_TYPE_TOP_K;
extern const char *G_OP_TYPE_CAST;
extern const char *G_OP_TYPE_LOG;
extern const char *G_OP_TYPE_LOD_RESET;
extern const char *G_OP_TYPE_LESS_THAN;
extern const char *G_OP_TYPE_LOGICAL_AND;
extern const char *G_OP_TYPE_LOGICAL_OR;
extern const char *G_OP_TYPE_LOGICAL_NOT;
extern const char *G_OP_TYPE_LOGICAL_XOR;
extern const char *G_OP_TYPE_WRITE_TO_ARRAY;
extern const char *G_OP_TYPE_READ_FROM_ARRAY;
extern const char *G_OP_TYPE_IS_EMPTY;
extern const char *G_OP_TYPE_INCREMENT;

extern const char *G_OP_TYPE_QUANTIZE;
extern const char *G_OP_TYPE_DEQUANTIZE;
extern const char *G_OP_TYPE_FUSION_DEQUANT_BN;
extern const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN;
extern const char *G_OP_TYPE_FUSION_DEQUANT_BN_RELU;
extern const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU;
extern const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_QUANT;
extern const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU_QUANT;

extern const char *G_OP_TYPE_TANH;
extern const char *G_OP_TYPE_FUSION_DECONV_RELU;

extern const char *G_OP_TYPE_FUSION_DECONV_ADD;
extern const char *G_OP_TYPE_FUSION_DECONV_ADD_RELU;

extern const char *G_OP_TYPE_SEQUENCE_EXPAND;
extern const char *G_OP_TYPE_SEQUENCE_POOL;
extern const char *G_OP_TYPE_SEQUENCE_SOFTMAX;

extern const char *G_OP_TYPE_SLICE;
extern const char *G_OP_TYPE_ANCHOR_GENERATOR;
extern const char *G_OP_TYPE_GENERATE_PROPOSALS;
extern const char *G_OP_TYPE_PSROI_POOL;
extern const char *G_OP_TYPE_ROIALIGN_POOL;
extern const char *G_OP_TYPE_ROI_PERSPECTIVE;
extern const char *G_OP_TYPE_PAD2D;
extern const char *G_OP_TYPE_FUSION_DECONV_ADD_BN_RELU;
extern const char *G_OP_TYPE_FUSION_DECONV_ADD_BN;
extern const char *G_OP_TYPE_FUSION_DECONV_BN_RELU;

extern const char *G_OP_TYPE_PAD2D;

extern std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    op_input_output_key;

typedef std::map<std::string, std::vector<std::string>> VariableNameMap;

}  // namespace paddle_mobile
