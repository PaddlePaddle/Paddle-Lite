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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef LITE_ON_TINY_PUBLISH
#include "lite/api/cxx_api.h"
#include "lite/api/opt_base.h"
#endif

#include "lite/api/light_api.h"
#include "lite/api/paddle_api.h"
#include "lite/core/tensor.h"

namespace py = pybind11;

namespace paddle {
namespace lite {
namespace pybind {

using lite_api::Tensor;
using lite_api::CxxConfig;
using lite_api::MobileConfig;
using lite_api::PowerMode;
using lite_api::TargetType;
using lite_api::PrecisionType;
using lite_api::DataLayoutType;
using lite_api::Place;
using lite_api::MLUCoreVersion;
using lite::LightPredictorImpl;
using lite_api::OptBase;

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
      .def("set_optimize_out", &OptBase::SetOptimizeOut)
      .def("set_model_type", &OptBase::SetModelType)
      .def("record_model_info", &OptBase::RecordModelInfo)
      .def("set_passes_internal", &OptBase::SetPassesInternal)
      .def("run", &OptBase::Run)
      .def("run_optimize", &OptBase::RunOptimize)
      .def("help", &OptBase::PrintHelpInfo)
      .def("executablebin_help", &OptBase::PrintExecutableBinHelpInfo)
      .def("print_supported_ops", &OptBase::PrintSupportedOps)
      .def("display_kernels_info", &OptBase::DisplayKernelsInfo)
      .def("print_all_ops", &OptBase::PrintAllOps)
      .def("check_if_model_supported", &OptBase::CheckIfModelSupported);
}
#endif
static void BindLiteLightPredictor(py::module *m);
static void BindLiteCxxConfig(py::module *m);
static void BindLiteMobileConfig(py::module *m);
static void BindLitePowerMode(py::module *m);
static void BindLitePlace(py::module *m);
static void BindLiteTensor(py::module *m);
static void BindLiteMLUCoreVersion(py::module *m);

void BindLiteApi(py::module *m) {
  BindLiteCxxConfig(m);
  BindLiteMobileConfig(m);
  BindLitePowerMode(m);
  BindLitePlace(m);
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
      .def("set_model_buffer", &CxxConfig::set_model_buffer)
      .def("set_passes_internal", &CxxConfig::set_passes_internal)
      .def("model_from_memory", &CxxConfig::model_from_memory);
#ifdef LITE_WITH_ARM
  cxx_config.def("set_threads", &CxxConfig::set_threads)
      .def("threads", &CxxConfig::threads)
      .def("set_power_mode", &CxxConfig::set_power_mode)
      .def("power_mode", &CxxConfig::power_mode);
#endif
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
      .def("set_model_from_buffer", &MobileConfig::set_model_from_buffer)
      .def("set_model_dir", &MobileConfig::set_model_dir)
      .def("model_dir", &MobileConfig::model_dir)
      .def("set_model_buffer", &MobileConfig::set_model_buffer)
      .def("model_from_memory", &MobileConfig::model_from_memory);
#ifdef LITE_WITH_ARM
  mobile_config.def("set_threads", &MobileConfig::set_threads)
      .def("threads", &MobileConfig::threads)
      .def("set_power_mode", &MobileConfig::set_power_mode)
      .def("power_mode", &MobileConfig::power_mode);
#endif
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

void BindLiteMLUCoreVersion(py::module *m) {
  py::enum_<MLUCoreVersion>(*m, "MLUCoreVersion")
      .value("LITE_MLU_220", MLUCoreVersion::MLU_220)
      .value("LITE_MLU_270", MLUCoreVersion::MLU_270);
}

void BindLitePlace(py::module *m) {
  // TargetType
  py::enum_<TargetType>(*m, "TargetType")
      .value("Host", TargetType::kHost)
      .value("X86", TargetType::kX86)
      .value("CUDA", TargetType::kCUDA)
      .value("ARM", TargetType::kARM)
      .value("OpenCL", TargetType::kOpenCL)
      .value("FPGA", TargetType::kFPGA)
      .value("NPU", TargetType::kNPU)
      .value("MLU", TargetType::kMLU)
      .value("RKNPU", TargetType::kRKNPU)
      .value("APU", TargetType::kAPU)
      .value("Any", TargetType::kAny);

  // PrecisionType
  py::enum_<PrecisionType>(*m, "PrecisionType")
      .value("FP16", PrecisionType::kFP16)
      .value("FP32", PrecisionType::kFloat)
      .value("INT8", PrecisionType::kInt8)
      .value("INT16", PrecisionType::kInt16)
      .value("INT32", PrecisionType::kInt32)
      .value("INT64", PrecisionType::kInt64)
      .value("BOOL", PrecisionType::kBool)
      .value("Any", PrecisionType::kAny);

  // DataLayoutType
  py::enum_<DataLayoutType>(*m, "DataLayoutType")
      .value("NCHW", DataLayoutType::kNCHW)
      .value("NHWC", DataLayoutType::kNHWC)
      .value("ImageDefault", DataLayoutType::kImageDefault)
      .value("ImageFolder", DataLayoutType::kImageFolder)
      .value("ImageNW", DataLayoutType::kImageNW)
      .value("Any", DataLayoutType::kAny);

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
      .def("shape", &Tensor::shape)
      .def("target", &Tensor::target)
      .def("precision", &Tensor::precision)
      .def("lod", &Tensor::lod)
      .def("set_lod", &Tensor::SetLoD);

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
#ifdef LITE_WITH_MLU
  tensor.def("set_uint8_data",
             [](Tensor &self,
                const std::vector<uint8_t> &data,
                TargetType type = TargetType::kHost) {
               if (type == TargetType::kHost) {
                 self.CopyFromCpu<uint8_t, TargetType::kHost>(data.data());
               }
             },
             py::arg("data"),
             py::arg("type") = TargetType::kHost);

  DO_GETTER_ONCE(uint8_t, "uint8_data");
#endif
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
      .def("run", &CxxPaddleApiImpl::Run)
      .def("get_version", &CxxPaddleApiImpl::GetVersion)
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
      .def("run", &LightPredictorImpl::Run)
      .def("get_version", &LightPredictorImpl::GetVersion);
}

}  // namespace pybind
}  // namespace lite
}  // namespace paddle
