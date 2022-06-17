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

#include "lite/api/python/pybind/pybind.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifndef LITE_ON_TINY_PUBLISH
#include "lite/api/cxx_api.h"
#include "lite/api/tools/opt_base.h"
#endif

#include "lite/api/light_api.h"
#include "lite/api/paddle_api.h"
#include "lite/api/python/pybind/tensor_py.h"
#include "lite/core/tensor.h"

namespace py = pybind11;

namespace paddle {
namespace lite {
namespace pybind {

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

using lite::LightPredictorImpl;
using lite_api::CxxConfig;
using lite_api::DataLayoutType;
using lite_api::MLUCoreVersion;
using lite_api::MobileConfig;
using lite_api::OptBase;
using lite_api::Place;
using lite_api::PowerMode;
using lite_api::PrecisionType;
using lite_api::TargetType;
using lite_api::CLTuneMode;
using lite_api::CLPrecisionType;
using lite_api::Tensor;
using lite_api::CxxModelBuffer;

#ifndef LITE_ON_TINY_PUBLISH
using lite::CxxPaddleApiImpl;
static void BindLiteCxxPredictor(py::module *m);
void BindLiteOpt(py::module *m) {
  py::class_<OptBase> opt_base(*m, "Opt");
  opt_base.def(py::init<>())
      .def("set_model_dir", &OptBase::SetModelDir)
      .def("set_modelset_dir", &OptBase::SetModelSetDir)
      .def("set_model_file", &OptBase::SetModelFile)
      .def("set_param_file", &OptBase::SetParamFile)
      .def("set_valid_places", &OptBase::SetValidPlaces)
      .def("enable_fp16", &OptBase::EnableFloat16)
      .def("set_optimize_out", &OptBase::SetOptimizeOut)
      .def("set_model_type", &OptBase::SetModelType)
      .def("set_quant_model", &OptBase::SetQuantModel)
      .def("set_quant_type", &OptBase::SetQuantType)
      .def("set_sparse_model", &OptBase::SetSparseModel)
      .def("set_sparse_threshold", &OptBase::SetSparseThreshold)
      .def("record_model_info", &OptBase::RecordModelInfo)
      .def("set_passes_internal", &OptBase::SetPassesInternal)
      .def("run", &OptBase::Run)
      .def("run_optimize", &OptBase::RunOptimize)
      .def("version", &OptBase::OptVersion)
      .def("help", &OptBase::PrintHelpInfo)
      .def("executablebin_help", &OptBase::PrintExecutableBinHelpInfo)
      .def("print_supported_ops", &OptBase::PrintSupportedOps)
      .def("display_kernels_info", &OptBase::DisplayKernelsInfo)
      .def("print_all_ops", &OptBase::PrintAllOps)
      .def("check_if_model_supported", &OptBase::CheckIfModelSupported)
      .def("print_all_ops_in_md_dormat",
           &OptBase::PrintAllSupportedOpsInMdformat)
      .def("visualize_optimized_nb_model", &OptBase::VisualizeOptimizedNBModel);
}
#endif
static void BindLiteLightPredictor(py::module *m);
static void BindLiteCxxConfig(py::module *m);
static void BindLiteMobileConfig(py::module *m);
static void BindLitePowerMode(py::module *m);
static void BindLitePlace(py::module *m);
static void BindLiteCLTuneMode(py::module *m);
static void BindLiteCLPrecisionType(py::module *m);
static void BindLiteTensor(py::module *m);
static void BindLiteMLUCoreVersion(py::module *m);

void BindLiteApi(py::module *m) {
  BindLiteCxxConfig(m);
  BindLiteMobileConfig(m);
  BindLitePowerMode(m);
  BindLitePlace(m);
  BindLiteCLTuneMode(m);
  BindLiteCLPrecisionType(m);
  BindLiteTensor(m);
  BindLiteMLUCoreVersion(m);
#ifndef LITE_ON_TINY_PUBLISH
  BindLiteCxxPredictor(m);
#endif
  BindLiteLightPredictor(m);
// Global helper methods
#ifndef LITE_ON_TINY_PUBLISH
  m->def("create_paddle_predictor",
         [](const CxxConfig &config) -> std::unique_ptr<CxxPaddleApiImpl> {
           auto x = std::unique_ptr<CxxPaddleApiImpl>(new CxxPaddleApiImpl());
           x->Init(config);
           return std::move(x);
         });
#endif
  m->def("create_paddle_predictor",
         [](const MobileConfig &config) -> std::unique_ptr<LightPredictorImpl> {
           auto x =
               std::unique_ptr<LightPredictorImpl>(new LightPredictorImpl());
           x->Init(config);
           return std::move(x);
         });
}

void BindLiteCxxConfig(py::module *m) {
  py::class_<CxxConfig> cxx_config(*m, "CxxConfig");

  cxx_config.def(py::init<>())
      .def("set_model_dir", &CxxConfig::set_model_dir)
      .def("model_dir", &CxxConfig::model_dir)
      .def("set_model_file", &CxxConfig::set_model_file)
      .def("model_file", &CxxConfig::model_file)
      .def("set_param_file", &CxxConfig::set_param_file)
      .def("param_file", &CxxConfig::param_file)
      .def("set_valid_places", &CxxConfig::set_valid_places)
      .def("set_model_buffer",
           (void (CxxConfig::*)(const char *, size_t, const char *, size_t)) &
               CxxConfig::set_model_buffer)
      .def("set_model_buffer",
           (void (CxxConfig::*)(std::shared_ptr<CxxModelBuffer>)) &
               CxxConfig::set_model_buffer)
      .def("set_passes_internal", &CxxConfig::set_passes_internal)
      .def("is_model_from_memory", &CxxConfig::is_model_from_memory)
      .def("add_discarded_pass", &CxxConfig::add_discarded_pass);
  cxx_config.def("set_threads", &CxxConfig::set_threads)
      .def("threads", &CxxConfig::threads)
      .def("set_power_mode", &CxxConfig::set_power_mode)
      .def("power_mode", &CxxConfig::power_mode);

  cxx_config
      .def("set_opencl_binary_path_name",
           &CxxConfig::set_opencl_binary_path_name)
      .def("set_opencl_tune", &CxxConfig::set_opencl_tune)
      .def("set_opencl_precision", &CxxConfig::set_opencl_precision);

  cxx_config
      .def("set_metal_use_mps",
           &CxxConfig::set_metal_use_mps,
           py::arg("flag") = true)
      .def("set_metal_use_memory_reuse", &CxxConfig::set_metal_use_memory_reuse)
      .def("set_metal_lib_path", &CxxConfig::set_metal_lib_path);

  cxx_config
      .def("set_nnadapter_device_names", &CxxConfig::set_nnadapter_device_names)
      .def("set_nnadapter_context_properties",
           &CxxConfig::set_nnadapter_context_properties)
      .def("set_nnadapter_model_cache_dir",
           &CxxConfig::set_nnadapter_model_cache_dir)
      .def("set_nnadapter_subgraph_partition_config_path",
           &CxxConfig::set_nnadapter_subgraph_partition_config_path)
      .def("set_nnadapter_mixed_precision_quantization_config_path",
           &CxxConfig::set_nnadapter_mixed_precision_quantization_config_path)
      .def("nnadapter_device_names", &CxxConfig::nnadapter_device_names)
      .def("nnadapter_context_properties",
           &CxxConfig::nnadapter_context_properties)
      .def("nnadapter_model_cache_dir", &CxxConfig::nnadapter_model_cache_dir)
      .def("nnadapter_subgraph_partition_config_path",
           &CxxConfig::nnadapter_subgraph_partition_config_path)
      .def("nnadapter_mixed_precision_quantization_config_path",
           &CxxConfig::nnadapter_mixed_precision_quantization_config_path);

#ifdef LITE_WITH_MLU
  cxx_config.def("set_mlu_core_version", &CxxConfig::set_mlu_core_version)
      .def("set_mlu_core_number", &CxxConfig::set_mlu_core_number)
      .def("set_mlu_input_layout", &CxxConfig::set_mlu_input_layout)
      .def("set_mlu_use_first_conv", &CxxConfig::set_mlu_use_first_conv)
      .def("set_mlu_first_conv_mean", &CxxConfig::set_mlu_first_conv_mean)
      .def("set_mlu_first_conv_std", &CxxConfig::set_mlu_first_conv_std);
#endif
}

// TODO(sangoly): Should MobileConfig be renamed to LightConfig ??
void BindLiteMobileConfig(py::module *m) {
  py::class_<MobileConfig> mobile_config(*m, "MobileConfig");

  mobile_config.def(py::init<>())
      .def("set_model_from_file", &MobileConfig::set_model_from_file)
      .def("set_model_from_buffer",
           overload_cast_<const std::string &>()(
               &MobileConfig::set_model_from_buffer))
      .def("set_model_dir", &MobileConfig::set_model_dir)
      .def("model_dir", &MobileConfig::model_dir)
      .def("set_model_buffer", &MobileConfig::set_model_buffer)
      .def("is_model_from_memory", &MobileConfig::is_model_from_memory);
#ifdef LITE_WITH_ARM
  mobile_config.def("set_threads", &MobileConfig::set_threads)
      .def("threads", &MobileConfig::threads)
      .def("set_power_mode", &MobileConfig::set_power_mode)
      .def("power_mode", &MobileConfig::power_mode);
#endif
  mobile_config
      .def("set_opencl_binary_path_name",
           &MobileConfig::set_opencl_binary_path_name)
      .def("set_opencl_tune", &MobileConfig::set_opencl_tune)
      .def("set_opencl_precision", &MobileConfig::set_opencl_precision);
  mobile_config
      .def("set_metal_use_mps",
           &MobileConfig::set_metal_use_mps,
           py::arg("flag") = true)
      .def("set_metal_use_memory_reuse",
           &MobileConfig::set_metal_use_memory_reuse)
      .def("set_metal_lib_path", &MobileConfig::set_metal_lib_path);
  mobile_config
      .def("set_nnadapter_device_names",
           &MobileConfig::set_nnadapter_device_names)
      .def("set_nnadapter_context_properties",
           &MobileConfig::set_nnadapter_context_properties)
      .def("set_nnadapter_model_cache_dir",
           &MobileConfig::set_nnadapter_model_cache_dir)
      .def("set_nnadapter_dynamic_shape_info",
           &MobileConfig::set_nnadapter_dynamic_shape_info)
      .def("set_nnadapter_model_cache_buffers",
           &MobileConfig::set_nnadapter_model_cache_buffers);
}

void BindLitePowerMode(py::module *m) {
  py::enum_<PowerMode>(*m, "PowerMode")
      .value("LITE_POWER_HIGH", PowerMode::LITE_POWER_HIGH)
      .value("LITE_POWER_LOW", PowerMode::LITE_POWER_LOW)
      .value("LITE_POWER_FULL", PowerMode::LITE_POWER_FULL)
      .value("LITE_POWER_NO_BIND", PowerMode::LITE_POWER_NO_BIND)
      .value("LITE_POWER_RAND_HIGH", PowerMode::LITE_POWER_RAND_HIGH)
      .value("LITE_POWER_RAND_LOW", PowerMode::LITE_POWER_RAND_LOW);
}

void BindLiteCLTuneMode(py::module *m) {
  py::enum_<CLTuneMode>(*m, "CLTuneMode")
      .value("CL_TUNE_NONE", CLTuneMode::CL_TUNE_NONE)
      .value("CL_TUNE_RAPID", CLTuneMode::CL_TUNE_RAPID)
      .value("CL_TUNE_NORMAL", CLTuneMode::CL_TUNE_NORMAL)
      .value("CL_TUNE_EXHAUSTIVE", CLTuneMode::CL_TUNE_EXHAUSTIVE);
}

void BindLiteCLPrecisionType(py::module *m) {
  py::enum_<CLPrecisionType>(*m, "CLPrecisionType")
      .value("CL_PRECISION_AUTO", CLPrecisionType::CL_PRECISION_AUTO)
      .value("CL_PRECISION_FP32", CLPrecisionType::CL_PRECISION_FP32)
      .value("CL_PRECISION_FP16", CLPrecisionType::CL_PRECISION_FP16);
}

void BindLiteMLUCoreVersion(py::module *m) {
  py::enum_<MLUCoreVersion>(*m, "MLUCoreVersion")
      .value("LITE_MLU_220", MLUCoreVersion::MLU_220)
      .value("LITE_MLU_270", MLUCoreVersion::MLU_270);
}

void BindLitePlace(py::module *m) {
  // TargetType
  py::enum_<TargetType>(*m, "TargetType")
      .value("Unk", TargetType::kUnk)
      .value("Host", TargetType::kHost)
      .value("X86", TargetType::kX86)
      .value("CUDA", TargetType::kCUDA)
      .value("ARM", TargetType::kARM)
      .value("OpenCL", TargetType::kOpenCL)
      .value("Any", TargetType::kAny)
      .value("FPGA", TargetType::kFPGA)
      .value("NPU", TargetType::kNPU)
      .value("XPU", TargetType::kXPU)
      .value("BM", TargetType::kBM)
      .value("MLU", TargetType::kMLU)
      .value("RKNPU", TargetType::kRKNPU)
      .value("APU", TargetType::kAPU)
      .value("HUAWEI_ASCEND_NPU", TargetType::kHuaweiAscendNPU)
      .value("IMAGINATION_NNA", TargetType::kImaginationNNA)
      .value("INTEL_FPGA", TargetType::kIntelFPGA)
      .value("Metal", TargetType::kMetal)
      .value("NNAdapter", TargetType::kNNAdapter);

  // PrecisionType
  py::enum_<PrecisionType>(*m, "PrecisionType")
      .value("Unk", PrecisionType::kUnk)
      .value("FP32", PrecisionType::kFloat)
      .value("INT8", PrecisionType::kInt8)
      .value("INT32", PrecisionType::kInt32)
      .value("Any", PrecisionType::kAny)
      .value("FP16", PrecisionType::kFP16)
      .value("BOOL", PrecisionType::kBool)
      .value("INT64", PrecisionType::kInt64)
      .value("INT16", PrecisionType::kInt16)
      .value("UINT8", PrecisionType::kUInt8)
      .value("FP64", PrecisionType::kFP64);

  // DataLayoutType
  py::enum_<DataLayoutType>(*m, "DataLayoutType")
      .value("Unk", DataLayoutType::kUnk)
      .value("NCHW", DataLayoutType::kNCHW)
      .value("Any", DataLayoutType::kAny)
      .value("NHWC", DataLayoutType::kNHWC)
      .value("ImageDefault", DataLayoutType::kImageDefault)
      .value("ImageFolder", DataLayoutType::kImageFolder)
      .value("ImageNW", DataLayoutType::kImageNW)
      .value("MetalTexture2DArray", DataLayoutType::kMetalTexture2DArray)
      .value("MetalTexture2D", DataLayoutType::kMetalTexture2D);

  // Place
  py::class_<Place>(*m, "Place")
      .def(py::init<TargetType, PrecisionType, DataLayoutType, int16_t>(),
           py::arg("target"),
           py::arg("percision") = PrecisionType::kFloat,
           py::arg("layout") = DataLayoutType::kNCHW,
           py::arg("device") = 0)
      .def("is_valid", &Place::is_valid);
}

void BindLiteTensor(py::module *m) {
  auto data_size_func = [](const std::vector<int64_t> &shape) -> int64_t {
    int64_t res = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      res *= shape[i];
    }
    return res;
  };

  py::class_<Tensor> tensor(*m, "Tensor");

  tensor.def("resize", &Tensor::Resize)
      .def("numpy", [](Tensor &self) { return TensorToPyArray(self); })
      .def("shape", &Tensor::shape)
      .def("target", &Tensor::target)
      .def("precision", &Tensor::precision)
      .def("lod", &Tensor::lod)
      .def("set_lod", &Tensor::SetLoD)
      .def("from_numpy",
           SetTensorFromPyArray,
           py::arg("array"),
           py::arg("place") = TargetType::kHost);

#define DO_GETTER_ONCE(data_type__, name__)                           \
  tensor.def(#name__, [=](Tensor &self) -> std::vector<data_type__> { \
    std::vector<data_type__> data;                                    \
    auto shape = self.shape();                                        \
    int64_t num = data_size_func(shape);                              \
    data.resize(num);                                                 \
    self.CopyToCpu<data_type__>(data.data());                         \
    return data;                                                      \
  });

#define DO_SETTER_ONCE(data_type__, name__)                              \
  tensor.def(                                                            \
      #name__,                                                           \
      [](Tensor &self,                                                   \
         const std::vector<data_type__> &data,                           \
         TargetType type = TargetType::kHost) {                          \
        if (type == TargetType::kHost || type == TargetType::kARM) {     \
          self.CopyFromCpu<data_type__, TargetType::kHost>(data.data()); \
        } else if (type == TargetType::kCUDA) {                          \
          self.CopyFromCpu<data_type__, TargetType::kCUDA>(data.data()); \
        }                                                                \
      },                                                                 \
      py::arg("data"),                                                   \
      py::arg("type") = TargetType::kHost);

#define DATA_GETTER_SETTER_ONCE(data_type__, name__) \
  DO_SETTER_ONCE(data_type__, set_##name__##_data)   \
  DO_GETTER_ONCE(data_type__, name__##_data)

  DATA_GETTER_SETTER_ONCE(int8_t, int8);
  DATA_GETTER_SETTER_ONCE(uint8_t, uint8);
  DATA_GETTER_SETTER_ONCE(int32_t, int32);
  DATA_GETTER_SETTER_ONCE(float, float);
#undef DO_GETTER_ONCE
#undef DO_SETTER_ONCE
#undef DATA_GETTER_SETTER_ONCE
}

#ifndef LITE_ON_TINY_PUBLISH
void BindLiteCxxPredictor(py::module *m) {
  py::class_<CxxPaddleApiImpl>(*m, "CxxPredictor")
      .def(py::init<>())
      .def("get_input", &CxxPaddleApiImpl::GetInput)
      .def("get_output", &CxxPaddleApiImpl::GetOutput)
      .def("get_output_names", &CxxPaddleApiImpl::GetOutputNames)
      .def("get_input_names", &CxxPaddleApiImpl::GetInputNames)
      .def("get_input_by_name", &CxxPaddleApiImpl::GetInputByName)
      .def("get_output_by_name", &CxxPaddleApiImpl::GetOutputByName)
      .def("run", &CxxPaddleApiImpl::Run)
      .def("get_version", &CxxPaddleApiImpl::GetVersion)
      .def("save_optimized_pb_model",
           [](CxxPaddleApiImpl &self, const std::string &output_dir) {
             self.SaveOptimizedModel(output_dir,
                                     lite_api::LiteModelType::kProtobuf);
           })
      .def("save_optimized_model",
           [](CxxPaddleApiImpl &self, const std::string &output_dir) {
             self.SaveOptimizedModel(output_dir,
                                     lite_api::LiteModelType::kNaiveBuffer);
           });
}
#endif

void BindLiteLightPredictor(py::module *m) {
  py::class_<LightPredictorImpl>(*m, "LightPredictor")
      .def(py::init<>())
      .def("get_input", &LightPredictorImpl::GetInput)
      .def("get_output", &LightPredictorImpl::GetOutput)
      .def("get_input_names", &LightPredictorImpl::GetInputNames)
      .def("get_output_names", &LightPredictorImpl::GetOutputNames)
      .def("get_input_by_name", &LightPredictorImpl::GetInputByName)
      .def("get_output_by_name", &LightPredictorImpl::GetOutputByName)
      .def("run", &LightPredictorImpl::Run)
      .def("get_version", &LightPredictorImpl::GetVersion);
}

}  // namespace pybind
}  // namespace lite
}  // namespace paddle
