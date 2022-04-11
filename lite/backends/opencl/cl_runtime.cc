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

#include "lite/backends/opencl/cl_runtime.h"
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/opencl/utils/cache.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/version.h"
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

CLRuntime* CLRuntime::Global() {
  thread_local CLRuntime cl_runtime_;
  cl_runtime_.Init();
  return &cl_runtime_;
}

void CLRuntime::Flush(const int index) {
  if (is_cl_runtime_initialized_ && index % opencl_flush_period_ == 0 &&
      command_queue_ != nullptr) {
    command_queue_->flush();
  }
}

CLRuntime::~CLRuntime() {
#ifdef LITE_WITH_LOG
  LOG(INFO) << "is_cl_runtime_initialized_:" << is_cl_runtime_initialized_;
#endif
  if (is_cl_runtime_initialized_ == false) {
    return;
  }

  if (command_queue_ != nullptr) {
    command_queue_->flush();
    command_queue_->finish();
  }
  // For controlling the destruction order
  command_queue_.reset();
  context_.reset();
  device_.reset();
  platform_.reset();
  device_info_.clear();
}

bool CLRuntime::Init() {
#ifdef LITE_WITH_LOG
  VLOG(6) << "is_cl_runtime_initialized_:" << is_cl_runtime_initialized_;
#endif
  if (is_cl_runtime_initialized_) {
    return true;
  }

  bool opencl_lib_found = paddle::lite::CLWrapper::Global()->OpenclLibFound();
#ifdef LITE_WITH_LOG
  LOG(INFO) << "opencl_lib_found:" << opencl_lib_found;
#endif
  if (opencl_lib_found == false) {
    return false;
  }

  bool dlsym_success = paddle::lite::CLWrapper::Global()->DlsymSuccess();
#ifdef LITE_WITH_LOG
  LOG(INFO) << "dlsym_success:" << dlsym_success;
#endif
  if (dlsym_success == false) {
    return false;
  }

  bool is_platform_init = InitializePlatform();
#ifdef LITE_WITH_LOG
  LOG(INFO) << "is_platform_init:" << is_platform_init;
#endif
  if (is_platform_init == false) {
    return false;
  }

  bool is_device_init = InitializeDevice();
#ifdef LITE_WITH_LOG
  LOG(INFO) << "is_device_init:" << is_device_init;
#endif
  if (is_device_init == false) {
    return false;
  }

  if ((is_platform_init == true) && (is_device_init == true)) {
    is_platform_device_init_success_ = true;
    context_ = CreateContext();
    command_queue_ = CreateCommandQueue(context());
    is_cl_runtime_initialized_ = true;
#ifdef LITE_WITH_LOG
    LOG(INFO) << "set is_cl_runtime_initialized_ = true";
#endif
  }
  set_precision();
  return is_cl_runtime_initialized_;
}

cl::Platform& CLRuntime::platform() {
  CHECK(platform_ != nullptr) << "platform_ is not initialized!";
  return *platform_;
}

cl::Context& CLRuntime::context() {
  if (context_ == nullptr) {
    LOG(FATAL) << "context_ create failed, check whether context create "
                  "successfully in CreateContext!";
  }
  return *context_;
}

cl::Device& CLRuntime::device() {
  if (device_ == nullptr) {
    LOG(FATAL) << "device_ is not initialized!";
  }
  return *device_;
}

std::map<std::string, std::unique_ptr<cl::Program>>& CLRuntime::program_map() {
  return programs_;
}

cl::CommandQueue& CLRuntime::command_queue() {
  if (command_queue_ == nullptr) {
    LOG(FATAL) << "command_queue_ create failed, check whether command queue "
                  "create successfully in CreateCommandQueue!";
  }
  return *command_queue_;
}

cl::Program& CLRuntime::GetProgram(const std::string& file_name,
                                   const std::string& options) {
  /* -I +CLRuntime::Global()->cl_path() + "/cl_kernel"*/
  std::string build_option = options + " -cl-fast-relaxed-math -cl-mad-enable";
  if (build_option.find("CL_DTYPE_") == std::string::npos) {
    if (lite_api::CL_PRECISION_FP16 == get_precision()) {
      build_option += " -DCL_DTYPE_half ";
    } else {
      build_option += " -DCL_DTYPE_float -DCL_DTYPE_FLOAT_FORCE ";
    }
  }
#ifdef LITE_WITH_LOG
  VLOG(4) << "precision_: " << CLPrecisionTypeToStr(precision_);
  VLOG(4) << "OpenCL build_option: " << build_option;
#endif

  STL::stringstream program_key_ss;
  program_key_ss << file_name << build_option;
  std::string program_key = program_key_ss.str();

  // Build flow: cache -> precompiled binary -> source
  bool ret = CheckFromCache(program_key);
  if (!ret) {
    ret = CheckFromPrecompiledBinary(program_key, build_option);
    if (!ret) {
      ret = CheckFromSource(file_name, program_key, build_option);
    }
  }

  if (ret) {
    return *(programs_[program_key]);
  } else {
    LOG(FATAL) << "GetProgram failed, program_key: " << program_key;
  }
}

bool CLRuntime::CheckFromCache(const std::string& program_key) {
  auto iter = programs_.find(program_key);
  if (iter != programs_.end()) {
#ifdef LITE_WITH_LOG
    VLOG(3) << " --- program -> " << program_key
            << " has been built in cache --- ";
#endif
    return true;
  } else {
    return false;
  }
}

static auto remove_file = [](const std::string& bin_file) {
  if (remove(bin_file.c_str()) != 0) {
    LOG(FATAL) << "Cannot delete invalid precomplied OpenCL binary[" << bin_file
               << "]!";
  } else {
    LOG(INFO) << "Invalid precomplied OpenCL binary[" << bin_file
              << "] has been deleted!";
  }
};

bool CLRuntime::CheckFromPrecompiledBinary(const std::string& program_key,
                                           const std::string& build_option) {
  bool ret = false;
  bool delete_bin_flag = false;
  auto path_name = GetBinaryPathName();
  if (path_name.size() != 2) return ret;

  // find binary
  std::string bin_file = path_name.at(0) + "/" + path_name.at(1);
  std::string precision_option = (precision_ == lite_api::CL_PRECISION_FP16)
                                     ? "Precision: FP16; "
                                     : "Precision: FP32; ";

  if (programs_.empty()) {
    // Check whether binary exist.
    // `IsFileExists()` will return true if bin_file is a existing dir.
    if (!IsFileExists(bin_file)) {
      LOG(WARNING)
          << "There is no precompiled OpenCL binary[" << bin_file
          << "] in the given OpenCL binary path. "
             "Also please make sure the storage directory exist "
             "and you have Write&Read permission. Jump to build program "
             "from source.";
    } else {
      LOG(INFO) << "Load opencl kernel bin file: " << bin_file;
      // Note that deserialize will fail if bin_file is a existing dir.
      bool success = Deserialize(bin_file, &programs_precompiled_binary_);
      if (!success) {
        LOG(WARNING) << "Failed to deserialize kernel binary file:" << bin_file;
        return ret;
      }

      VLOG(3) << "sn_key: " << sn_key_;
      VLOG(3) << "map size: " << programs_precompiled_binary_.size();
      for (auto& ins : programs_precompiled_binary_) {
        std::string prog_key = ins.first;
        VLOG(3) << "\t map key: " << prog_key
                << "\t map value size: " << ins.second[0].size();
      }

      // check if the binary file is illegal and valid
      auto sn_iter = programs_precompiled_binary_.find(sn_key_);
      if (sn_iter == programs_precompiled_binary_.end()) {
        LOG(WARNING) << "The precompiled OpenCL binary[" << bin_file
                     << "] is illegal!";
        delete_bin_flag = true;
        del_tune_bin_flag_ = true;
        // Jump to build from source
      } else if (host::memcmp(((sn_iter->second)[0]).data(),
                              GetSN(precision_option).data(),
                              GetSN(precision_option).length())) {
        std::string sn_str(reinterpret_cast<char*>((sn_iter->second)[0].data()),
                           (sn_iter->second)[0].size());
        LOG(INFO) << "\nSN required: " << GetSN(precision_option)
                  << "\tsize: " << GetSN(precision_option).length()
                  << "\nSN in bin file: " << sn_str
                  << "\tsize: " << ((sn_iter->second)[0]).size();
        LOG(WARNING) << "The precompiled OpenCL binary[" << bin_file
                     << "] is invalid!";
        delete_bin_flag = true;
        del_tune_bin_flag_ = true;
        // Jump to build from source
      } else {
#ifdef LITE_WITH_LOG
        VLOG(3) << " --- begin read all precompiled programs from binary --- ";
#endif
        // loop all programs of the binary file
        cl_int status{CL_SUCCESS};
        for (auto& ins : programs_precompiled_binary_) {
          std::string prog_key = ins.first;
          if (prog_key == sn_key_) continue;  // skip sn_key

          cl::Program program(
              context(), {device()}, ins.second, nullptr, &status);
          CL_CHECK_FATAL_SOLID(status);
          auto pos_start = prog_key.find_first_of("-D");
          std::string options = prog_key.substr(pos_start);
          BuildProgram(&program, options);

          std::unique_ptr<cl::Program> ptr(new cl::Program(program));
          programs_[prog_key] = std::move(ptr);
        }

        auto it = programs_.find(program_key);
        if (it != programs_.end()) {
          VLOG(3) << " --- program -> " << program_key
                  << " has been built in binary --- ";
          gotten_bin_flag_ = true;
          ret = true;
        } else {
          delete_bin_flag = true;
          del_tune_bin_flag_ = true;
          // Jump to build from source
        }
      }
    }

    if (delete_bin_flag) {
      remove_file(bin_file);
      programs_precompiled_binary_.clear();
      programs_.clear();
    }
  } else if (gotten_bin_flag_) {
    // This case happened when model has updated. Bin file should be updated
    // accordingly.
    delete_bin_flag = true;
    del_tune_bin_flag_ = true;
    gotten_bin_flag_ = false;
    remove_file(bin_file);
  }

  return ret;
}

bool CLRuntime::CheckFromSource(const std::string& file_name,
                                const std::string& program_key,
                                const std::string& build_option) {
  auto ptr = CreateProgramFromSource(context(), file_name);
  auto program = ptr.get();
#ifdef LITE_WITH_LOG
  VLOG(3) << " --- begin build program from source -> " << program_key
          << " --- ";
#endif
  BuildProgram(program, build_option);
  programs_[program_key] = std::move(ptr);

  return true;
}

std::unique_ptr<cl::Program> CLRuntime::CreateProgramFromSource(
    const cl::Context& context, std::string file_name) {
  auto cl_file = opencl_kernels_files.find(file_name);
  std::string content(cl_file->second.begin(), cl_file->second.end());
  cl::Program::Sources sources;
  sources.push_back(content);
  auto prog =
      std::unique_ptr<cl::Program>(new cl::Program(context, sources, &status_));
#ifdef LITE_WITH_LOG
  VLOG(4) << "OpenCL kernel file name: " << file_name;
  VLOG(4) << "Program source size: " << content.size();
#endif
  CL_CHECK_FATAL_SOLID(status_);
  return std::move(prog);
}

bool CLRuntime::BuildProgram(cl::Program* program, const std::string& options) {
  status_ = program->build({device()}, options.c_str());
  CL_CHECK_ERROR(status_);

  if (status_ != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(FATAL) << "Program build error: " << log;
    }
    return false;
  }

  return true;
}

void CLRuntime::SaveProgram() {
  if (binary_path_name_.empty()) return;
  std::string binary_file =
      binary_path_name_.at(0) + "/" + binary_path_name_.at(1);
  if (IsFileExists(binary_file)) {
    LOG(INFO) << "OpenCL Program existed:" << binary_file;
  } else {
    for (auto& program_id : programs_) {
      // Keep built program binary
      if (binary_path_name_.size() == 2) {
        cl_int status{CL_SUCCESS};
        // 1. Query binary (PTX file) size
        size_t bin_size;
        cl::Program program = *(program_id.second);
        status = program.getInfo(CL_PROGRAM_BINARY_SIZES, &bin_size);
        CL_CHECK_FATAL_SOLID(status);
        // 2. Read binary (PTX file) to memory buffer
        cl::Program::Binaries binary;
        binary.resize(1);
        binary[0].resize(bin_size);
        auto buf = binary[0].data();
        status = program.getInfo(CL_PROGRAM_BINARIES, &buf);
        CL_CHECK_FATAL_SOLID(status);
        programs_precompiled_binary_[program_id.first] = binary;
#ifdef LITE_WITH_LOG
        VLOG(3) << " --- binary size: " << bin_size << " ---";
#endif
        if (programs_precompiled_binary_.find(sn_key_) ==
            programs_precompiled_binary_.end()) {
          // add identifier
          std::string precision_option =
              (precision_ == lite_api::CL_PRECISION_FP16) ? "Precision: FP16; "
                                                          : "Precision: FP32; ";
          std::string sn = GetSN(precision_option);
          std::vector<unsigned char> sn_info(sn.data(), sn.data() + sn.size());
          programs_precompiled_binary_[sn_key_] = {sn_info};
        }
      }
    }
    bool ret = Serialize(binary_file, programs_precompiled_binary_);
    if (!ret) {
      LOG(WARNING) << "Serialize failed for opencl binary_file:" << binary_file;
    }
#ifdef LITE_WITH_LOG
    if (programs_precompiled_binary_.find(sn_key_) !=
        programs_precompiled_binary_.end()) {
      std::string sn_str(reinterpret_cast<char*>(
                             programs_precompiled_binary_[sn_key_][0].data()),
                         programs_precompiled_binary_[sn_key_][0].size());
      LOG(INFO) << "SN stored: " << sn_str;
    }
    LOG(INFO) << "Programs have been serialized to disk successfully. File: "
              << binary_file;
#endif
  }
}

void CLRuntime::SaveTuned() {
  if (tuned_path_name_.empty() || auto_tune() == lite_api::CL_TUNE_NONE) return;
  std::string tuned_file =
      tuned_path_name_.at(0) + "/" + tuned_path_name_.at(1);
  if (IsFileExists(tuned_file) && del_tune_bin_flag_) {
    remove_file(tuned_file);
  }
  if (IsFileExists(tuned_file)) {
    LOG(INFO) << "OpenCL Tuned file existed:" << tuned_file;
  } else {
    bool ret = Serialize(tuned_file, tuned_lwss_map_);
    if (!ret) {
      LOG(WARNING) << "Serialize failed for opencl tuned_file:" << tuned_file;
    }
    LOG(INFO) << "Tuned file have been serialized to disk successfully: "
              << tuned_file;
  }
}

// binary
bool CLRuntime::Serialize(
    const std::string file_name,
    const std::map<std::string, cl::Program::Binaries>& map_data) {
  fbs::opencl::Cache cache{map_data};
  std::vector<uint8_t> buffer;
  cache.CopyDataToBuffer(&buffer);

  WriteFile<uint8_t>(file_name, buffer);
  return true;
}

bool CLRuntime::Deserialize(
    const std::string file_name,
    std::map<std::string, cl::Program::Binaries>* map_ptr) {
  std::vector<uint8_t> buffer;
  ReadFile<uint8_t>(file_name, &buffer);

  fbs::opencl::Cache cache{buffer};
  *map_ptr = cache.GetBinaryMap();
  return true;
}

// tuned param
bool CLRuntime::Serialize(
    const std::string file_name,
    const std::map<std::string, std::vector<int>>& map_data) {
  fbs::opencl::TuneCache cache{map_data};
  std::vector<int> buffer;
  cache.CopyDataToBuffer(&buffer);

  WriteFile<int>(file_name, buffer);
  return true;
}

bool CLRuntime::Deserialize(const std::string file_name,
                            std::map<std::string, std::vector<int>>* map_ptr) {
  std::vector<int> buffer;
  ReadFile<int>(file_name, &buffer);

  fbs::opencl::TuneCache cache{buffer};
  *map_ptr = cache.GetBinaryMap();
  return true;
}

std::string CLRuntime::GetSN(const std::string options) {
  // Identifier info(Serial Number) for each binary file: lite version,
  // build options, platform info, device version, driver version
  STL::stringstream sn_ss;

  const std::string aarch =
#if defined(__aarch64__)
      "android_armv8";
#else
      "android_armv7";
#endif
#if defined(_WIN64)
  "win64";
#elif defined(_WIN32)
  "win32";
#endif

  const std::string aarch_info = aarch + "; ";
  const std::string lite_version = lite::version() + "; ";
  const std::string platform_info =
      platform_->getInfo<CL_PLATFORM_NAME>() + ", " +
      platform_->getInfo<CL_PLATFORM_PROFILE>() + "; ";
  const std::string device_version =
      device_->getInfo<CL_DEVICE_VERSION>() + "; ";
  const std::string driver_version =
      device_->getInfo<CL_DRIVER_VERSION>() + "; ";
  const std::string place_holder{"place_holder"};
  sn_ss << aarch_info << lite_version << options << platform_info
        << device_version << driver_version << place_holder;
  return sn_ss.str();
}

std::unique_ptr<cl::UserEvent> CLRuntime::CreateEvent(
    const cl::Context& context) {
  auto event =
      std::unique_ptr<cl::UserEvent>(new cl::UserEvent(context, &status_));
  CL_CHECK_FATAL_SOLID(status_);
  return std::move(event);
}

bool CLRuntime::InitializePlatform() {
  std::vector<cl::Platform> all_platforms;
  status_ = cl::Platform::get(&all_platforms);
  // has return status do not exit here when release
  CL_CHECK_ERROR(status_);
  if (all_platforms.empty()) {
    LOG(ERROR) << "No OpenCL platform found!";
    return false;
  }
  platform_ = std::make_shared<cl::Platform>();
  *platform_ = all_platforms[0];
  const std::string extensions = platform_->getInfo<CL_PLATFORM_EXTENSIONS>();
  LOG(INFO) << "Platform extension: " << extensions;
  return true;
}

OpenCLVersion CLRuntime::ParseDeviceVersion(const std::string& device_version) {
  // OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <vendor-specific information>
  auto words = Split<std::string>(device_version, std::string{" "});
  if (words[1] == "2.1") {
    return OpenCLVersion::CL_VER_2_1;
  } else if (words[1] == "2.0") {
    return OpenCLVersion::CL_VER_2_0;
  } else if (words[1] == "1.2") {
    return OpenCLVersion::CL_VER_1_2;
  } else if (words[1] == "1.1") {
    return OpenCLVersion::CL_VER_1_1;
  } else if (words[1] == "1.0") {
    return OpenCLVersion::CL_VER_1_0;
  } else {
    LOG(ERROR) << "Do not support OpenCL version: " << words[1];
    return OpenCLVersion::CL_VER_UNKNOWN;
  }
}

GpuType CLRuntime::ParseGpuTypeFromDeviceName(std::string device_name) {
  const std::string kMALI_PATTERN_STR = "Mali";
  const std::string kADRENO_PATTERN_STR = "QUALCOMM Adreno(TM)";
  const std::string kPOWERVR_PATTERN_STR = "PowerVR";
  const std::string kAPPLE_M1_PATTERN_STR = "Apple M1";
  std::string gpu_type_str = "";

  if (device_name == kADRENO_PATTERN_STR) {
    gpu_type_str = "adreno gpu";
    return GpuType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMALI_PATTERN_STR) != std::string::npos) {
    gpu_type_str = "mali gpu";
    return GpuType::ARM_MALI;
  } else if (device_name.find(kPOWERVR_PATTERN_STR) != std::string::npos) {
    gpu_type_str = "powerVR gpu";
    return GpuType::IMAGINATION_POWERVR;
  } else if (device_name.find(kAPPLE_M1_PATTERN_STR) != std::string::npos) {
    gpu_type_str = "appleM1 gpu";
    return GpuType::APPLE_M1;
  } else {
    gpu_type_str = "others gpu";
    return GpuType::UNKNOWN;
  }
#ifdef LITE_WITH_LOG
  LOG(INFO) << "gpu_type_str:" << gpu_type_str;
#endif
}

bool CLRuntime::InitializeDevice() {
  // initialized without valid opencl device
  if (device_info_.size() > 0 && device_info_.size() <= 2) {
    return false;
  }
  // initialized with valid opencl device
  if (device_info_.size() > 2) {
    return true;
  }
  device_info_["PLACEHOLDER"] = 1;
  // ===================== BASIC =====================
  // CL_DEVICE_TYPE_GPU
  // CL_DEVICE_NAME
  // CL_DEVICE_SUPPORT
  // CL_DEVICE_MAX_COMPUTE_UNITS
  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  std::vector<cl::Device> all_devices;
  status_ = platform_->getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  // for is_opencl_valid_api .  do not exit here...
  CL_CHECK_ERROR(status_);
  if (all_devices.empty()) {
    LOG(ERROR)
        << "No available OpenCL GPU device found!, Try CPU Device instead!";
    status_ = platform_->getDevices(CL_DEVICE_TYPE_CPU, &all_devices);
    CL_CHECK_ERROR(status_);
    if (all_devices.empty()) {
      return false;
    }
  }
  device_ = std::make_shared<cl::Device>();
  *device_ = all_devices[0];

  auto device_name = device_->getInfo<CL_DEVICE_NAME>();
  LOG(INFO) << "Using device: " << device_name;
  gpu_type_ = ParseGpuTypeFromDeviceName(device_name);

  cl_device_type device_type = device_->getInfo<CL_DEVICE_TYPE>();
  auto device_type_to_str = [](cl_device_type t) -> std::string {
    std::string t_str{""};
    switch (t) {
      case CL_DEVICE_TYPE_CPU:
        t_str = "CPU";
        break;
      case CL_DEVICE_TYPE_GPU:
        t_str = "GPU";
        break;
      case CL_DEVICE_TYPE_ACCELERATOR:
        t_str = "Accelerator";
        break;
      case CL_DEVICE_TYPE_DEFAULT:
        t_str = "Default";
        break;
      default:
        t_str = "Unknown";
    }
    return t_str;
  };

  auto device_version = device_->getInfo<CL_DEVICE_VERSION>();
  LOG(INFO) << "CL_DEVICE_VERSION:" << device_version;
  auto opencl_version = ParseDeviceVersion(device_version);
  if (opencl_version == OpenCLVersion::CL_VER_UNKNOWN) {
    LOG(ERROR) << "Parse device version[" << device_version << "] failed!";
  }
  device_info_["CL_DEVICE_VERSION"] = opencl_version;

  LOG(INFO) << "device_type:" << device_type_to_str(device_type);
  device_info_["CL_DEVICE_TYPE"] = device_type;

  auto max_units = device_->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  LOG(INFO) << "The chosen device has " << max_units << " compute units.";
  device_info_["CL_DEVICE_MAX_COMPUTE_UNITS"] = max_units;

  auto max_clock_freq = device_->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  LOG(INFO) << "CL_DEVICE_MAX_CLOCK_FREQUENCY:" << max_clock_freq;
  device_info_["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = max_clock_freq;

  // ===================== MEMORY =====================
  // CL_DEVICE_LOCAL_MEM_SIZE
  // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
  // CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
  // CL_DEVICE_GLOBAL_MEM_SIZE
  auto local_mem_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()) / 1024;
  LOG(INFO) << "The local memory size of the chosen device is " << local_mem_kb
            << " KB.";
  device_info_["CL_DEVICE_LOCAL_MEM_SIZE_KB"] = local_mem_kb;

  auto global_mem_cache_size_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE(KB):"
            << global_mem_cache_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_CACHE_SIZE_KB"] = global_mem_cache_size_kb;

  auto global_mem_cacheline_size_kb =
      static_cast<float>(
          device_->getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE(KB):"
            << global_mem_cacheline_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE_KB"] =
      global_mem_cacheline_size_kb;

  auto global_mem_size_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) / 1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_SIZE(KB):" << global_mem_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_SIZE_KB"] = global_mem_size_kb;

  // ===================== WORK_GROUP =====================
  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  auto max_work_group_size = device_->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  LOG(INFO) << "CL_DEVICE_MAX_WORK_GROUP_SIZE:" << max_work_group_size;
  device_info_["CL_DEVICE_MAX_WORK_GROUP_SIZE"] = max_work_group_size;

  auto max_dims_num = device_->getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
  LOG(INFO) << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:" << max_dims_num;
  device_info_["CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"] = max_dims_num;

  auto max_work_item_sizes = device_->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (size_t i = 0; i < max_work_item_sizes.size(); ++i) {
    LOG(INFO) << "max_work_item_sizes[" << i << "]:" << max_work_item_sizes[i];
    std::string dim_key = "CL_DEVICE_MAX_WORK_ITEM_SIZES_" + std::to_string(i);
    device_info_[dim_key] = max_work_item_sizes[i];
  }

  // ===================== BUFFER =====================
  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  auto max_constant_buffer_size_kb =
      static_cast<float>(
          device_->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:"
            << max_constant_buffer_size_kb;
  device_info_["CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE"] =
      max_constant_buffer_size_kb;

  // ===================== IMAGE =====================
  // CL_DEVICE_IMAGE_SUPPORT
  // CL_DEVICE_IMAGE2D_MAX_HEIGHT
  // CL_DEVICE_IMAGE2D_MAX_WIDTH
  auto image_support = device_->getInfo<CL_DEVICE_IMAGE_SUPPORT>();
  if (image_support) {
    LOG(INFO) << "The chosen device supports image processing.";
    device_info_["CL_DEVICE_IMAGE_SUPPORT"] = 1;

    auto image2d_max_height = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
    LOG(INFO) << "CL_DEVICE_IMAGE2D_MAX_HEIGHT:" << image2d_max_height;
    device_info_["CL_DEVICE_IMAGE2D_MAX_HEIGHT"] = image2d_max_height;

    auto image2d_max_width = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    LOG(INFO) << "CL_DEVICE_IMAGE2D_MAX_WIDTH:" << image2d_max_width;
    device_info_["CL_DEVICE_IMAGE2D_MAX_WIDTH"] = image2d_max_width;
  } else {
    LOG(ERROR) << "The chosen device doesn't support image processing!";
    device_info_["CL_DEVICE_IMAGE_SUPPORT"] = 0;
    return false;
  }

  // ===================== OTHERS / EXTENSION / VERSION =====================
  // CL_DEVICE_EXTENSIONS
  // CL_DEVICE_ADDRESS_BITS
  auto ext_data = device_->getInfo<CL_DEVICE_EXTENSIONS>();
  VLOG(4) << "The extensions supported by this device: " << ext_data;
  if (ext_data.find("cl_khr_fp16") != std::string::npos) {
    LOG(INFO) << "The chosen device supports the half data type.";
    device_info_["CL_DEVICE_EXTENSIONS_FP16"] = 1;
  } else {
    LOG(INFO) << "The chosen device doesn't support the half data type!";
    device_info_["CL_DEVICE_EXTENSIONS_FP16"] = 0;
  }

  auto address_bits = device_->getInfo<CL_DEVICE_ADDRESS_BITS>();
  LOG(INFO) << "CL_DEVICE_ADDRESS_BITS:" << address_bits;
  device_info_["CL_DEVICE_ADDRESS_BITS"] = address_bits;

  auto driver_version = device_->getInfo<CL_DRIVER_VERSION>();
  LOG(INFO) << "CL_DRIVER_VERSION:" << driver_version;

#ifdef LITE_WITH_LOG
  VLOG(3) << "device_info_.size():" << device_info_.size();
  for (auto i : device_info_) {
    VLOG(3) << ">>> " << i.first << " " << i.second;
  }
#endif

  return true;
}

std::map<std::string, size_t>& CLRuntime::GetDeviceInfo() {
  InitializeDevice();
  return device_info_;
}

GpuType& CLRuntime::GetGpuType() { return gpu_type_; }

void CLRuntime::GetAdrenoContextProperties(
    std::vector<cl_context_properties>* properties,
    GPUPerfMode gpu_perf_mode,
    GPUPriorityLevel gpu_priority_level) {
  if (properties == nullptr) {
    LOG(ERROR) << "cl_context_properties is nullptr";
    return;
  }
  properties->reserve(5);
  switch (gpu_perf_mode) {
    case GPUPerfMode::PERF_LOW:
      LOG(INFO) << "GPUPerfMode::PERF_LOW";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_LOW_QCOM);
      break;
    case GPUPerfMode::PERF_NORMAL:
      LOG(INFO) << "GPUPerfMode::PERF_NORMAL";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_NORMAL_QCOM);
      break;
    case GPUPerfMode::PERF_HIGH:
      LOG(INFO) << "GPUPerfMode::PERF_HIGH";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_HIGH_QCOM);
      break;
    default:
      break;
  }
  switch (gpu_priority_level) {
    case GPUPriorityLevel::PRIORITY_LOW:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_LOW";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_LOW_QCOM);
      break;
    case GPUPriorityLevel::PRIORITY_NORMAL:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_NORMAL";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
      break;
    case GPUPriorityLevel::PRIORITY_HIGH:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_HIGH";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  // The properties list should be terminated with 0
  properties->push_back(0);
}

uint64_t CLRuntime::GetMaxWorkGroupSize(const cl::Kernel& kernel) {
  uint64_t max_workgroup_size = 0;
  int ret = kernel.getWorkGroupInfo(
      *device_, CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
  if (ret != 0) max_workgroup_size = 0;
  return max_workgroup_size;
}

void CLRuntime::set_auto_tune(lite_api::CLTuneMode tune_mode,
                              const std::string& path,
                              const std::string& name,
                              size_t lws_repeats) {
  auto_tune_ = tune_mode;
  auto device_name = CLRuntime::Global()->device().getInfo<CL_DEVICE_NAME>();
  if (device_name.find("Mali-T860") != std::string::npos) {
    auto_tune_ = lite_api::CL_TUNE_NONE;
  }
  lws_repeats_ = lws_repeats;

  tuned_path_name_.clear();
  tuned_path_name_.push_back(path);
  tuned_path_name_.push_back(name);
  const std::string tuned_file =
      tuned_path_name_.at(0) + "/" + tuned_path_name_.at(1);
  LOG(INFO) << "tuned_file:" << tuned_file;
  if (IsFileExists(tuned_file) && auto_tune() != lite_api::CL_TUNE_NONE) {
    LOG(INFO) << "Load tuned file: " << tuned_file;
    bool status = Deserialize(tuned_file, &tuned_lwss_map_);
    if (!status) {
      LOG(WARNING) << "Failed to deserialize tuned file:" << tuned_file;
    }
    have_tune_file_flag_ = true;
  } else {
    LOG(WARNING) << "Not found tuned file:" << tuned_file;
  }
  command_queue_ = CreateCommandQueue(context());
}

bool CLRuntime::HasTunedLocalWorkSizeMap(const std::string& key,
                                         std::vector<int>* tuned_value) {
  bool has = false;
  auto it = tuned_lwss_map_.find(key);
  if (it != tuned_lwss_map_.end()) {
    *tuned_value = it->second;
    has = true;
  }
  return has;
}

void CLRuntime::SetTunedLocalWorkSizeMap(const std::string& key,
                                         const std::vector<int>& tune_vct) {
  auto it = tuned_lwss_map_.find(key);
  if (it != tuned_lwss_map_.end()) {
    auto lws_old = it->second;
    LOG(FATAL) << "===> found lws_old with same key, please add more detailed "
                  "info to key <==="
               << "\n lws_old:" << lws_old[0] << "," << lws_old[1] << ","
               << lws_old[2] << "\n lws_new:" << tune_vct[0] << ","
               << tune_vct[1] << "," << tune_vct[2];
  }
  tuned_lwss_map_.insert(
      std::pair<std::string, std::vector<int>>(key, tune_vct));
}

double CLRuntime::GetCommandTime(const cl::Event& event) {
  event.wait();
  auto start_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto stop_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return (stop_nanos - start_nanos) / 1000000.0;
}

double CLRuntime::GetQueuedTime(const cl::Event& event) {
  event.wait();
  return (event.getProfilingInfo<CL_PROFILING_COMMAND_START>() -
          event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) /
         1000000.0;
}

double CLRuntime::GetSubmitTime(const cl::Event& event) {
  event.wait();
  return (event.getProfilingInfo<CL_PROFILING_COMMAND_START>() -
          event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) /
         1000000.0;
}

}  // namespace lite
}  // namespace paddle
