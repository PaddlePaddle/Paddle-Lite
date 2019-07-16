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

#include "lite/model_parser/model_parser.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/variable.h"
#include "lite/model_parser/desc_apis.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"

namespace paddle {
namespace lite {

int SizeOfType(framework::proto::VarType::Type type) {
  using Type = framework::proto::VarType::Type;
  switch (static_cast<int>(type)) {
#define DO(desc, type)            \
  case Type::VarType_Type_##desc: \
    return sizeof(type);
    DO(BOOL, bool);
    DO(FP16, float);
    DO(FP32, float);
    DO(INT8, int8_t);
    DO(INT32, int);
    DO(INT64, int64_t);
#undef DO
    default:
      LOG(FATAL) << "unknown data type " << type;
  }
  return -1;
}

void TensorFromStream(std::istream &is, lite::Tensor *tensor) {
  using Type = framework::proto::VarType::Type;
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK_EQ(version, 0U) << "Only version 0 is supported";
  // read tensor desc
  framework::proto::VarType::TensorDesc desc;
  {
    // int32_t size
    // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char *>(buf.get()), size);
    CHECK(desc.ParseFromArray(buf.get(), size)) << "Cannot parse tensor desc";
  }

  // read tensor
  std::vector<int64_t> dims_vec;
  std::copy(
      desc.dims().begin(), desc.dims().end(), std::back_inserter(dims_vec));
  lite::DDim dims(dims_vec);
  tensor->Resize(dims);
  void *buf;
  size_t size = tensor->dims().production() * SizeOfType(desc.data_type());
  // alllocate memory
  switch (static_cast<int>(desc.data_type())) {
#define DO(desc, type)                  \
  case Type::VarType_Type_##desc:       \
    buf = tensor->mutable_data<type>(); \
    break;
    // DO(BOOL, bool);
    DO(FP32, float);
    DO(INT8, int8_t);
    DO(INT16, int16_t);
    DO(INT32, int32_t);
    DO(INT64, int64_t);
#undef DO
    default:
      LOG(FATAL) << "unknown type " << desc.data_type();
  }

  is.read(static_cast<char *>(buf), size);
}

void LoadLoDTensor(std::istream &is, Variable *var) {
  auto *tensor = var->GetMutable<lite::Tensor>();
  uint32_t version{};
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  VLOG(3) << "model version " << version;

  // Load LoD information
  uint64_t lod_level{};
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<uint64_t> tmp(size / sizeof(uint64_t));
    is.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(size));
    lod[i] = tmp;
  }

  TensorFromStream(is, tensor);
}

// TODO(Superjomn) support SelectedRows.

void ReadBinaryFile(const std::string &filename, std::string *contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "Cannot open file: " << filename;
  fin.seekg(0, std::ios::end);
  auto size = fin.tellg();
  contents->clear();
  contents->resize(size);
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<framework::proto::ProgramDesc> LoadProgram(
    const std::string &path) {
  std::string desc_str;
  ReadBinaryFile(path, &desc_str);
  std::unique_ptr<framework::proto::ProgramDesc> main_program(
      new framework::proto::ProgramDesc);
  main_program->ParseFromString(desc_str);
  return main_program;
}

void LoadParams(const std::string &path) {}

// Load directly to CPU, and latter transfer to other devices.
void LoadParam(const std::string &path, Variable *out) {
  std::ifstream fin(path, std::ios::binary);
  CHECK(fin.is_open()) << "failed to open file " << path;
  LoadLoDTensor(fin, out);
}

void LoadModel(const std::string &model_dir,
               Scope *scope,
               framework::proto::ProgramDesc *prog) {
  const std::string prog_path = model_dir + "/__model__";
  *prog = *LoadProgram(prog_path);

  auto main_block = prog->blocks(0);
  for (auto &var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable())
      continue;

    std::string file_path = model_dir + "/" + var.name();
    VLOG(4) << "reading weight " << var.name();

    std::ifstream file(file_path);
    switch (var.type().type()) {
      case framework::proto::VarType_Type_LOD_TENSOR:
        LoadLoDTensor(file, scope->Var(var.name()));
        break;
      default:
        CHECK(false) << "unknown weight type";
    }
  }
}

void TensorToStream(std::ostream &os, const lite::Tensor &tensor) {
  // the 1st field, uint32_t version
  constexpr uint32_t version = 0;
  os.write(reinterpret_cast<const char *>(&version), sizeof(version));

  {
    uint64_t size = tensor.lod().size();
    // the 2st field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : tensor.lod()) {
      size = each.size() * sizeof(each.front());
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }

  // There are two version fields in a LoDTensor.
  os.write(reinterpret_cast<const char *>(&version), sizeof(version));

  {  // the 2nd field, tensor description
    // int32_t  size
    // void*    protobuf message
    framework::proto::VarType::TensorDesc desc;
    // TODO(Superjomn) support other data types.
    desc.set_data_type(framework::proto::VarType_Type_FP32);
    auto dims = tensor.dims();
    auto *pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    auto dims_vec = dims.Vectorize();
    std::copy(dims_vec.begin(), dims_vec.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size = tensor.memory_size();
    CHECK_LT(size, std::numeric_limits<std::streamsize>::max())
        << "Index overflow when writing tensor";

#ifdef LITE_WITH_CUDA
    if (tensor.target() == TARGET(kCUDA)) {
      std::unique_ptr<char> tmp_buffer(new char[size]);
      TargetWrapperCuda::MemcpySync(tmp_buffer.get(),
                                    tensor.data<float>(),
                                    tensor.data_size(),
                                    IoDirection::DtoH);
      os.write(static_cast<const char *>(tmp_buffer.get()),
               static_cast<std::streamsize>(size));
    } else  // NOLINT
#endif      // LITE_WITH_CUDA
    {
      os.write(static_cast<const char *>(tensor.data<void>()),
               static_cast<std::streamsize>(size));
    }
  }
}

void SerializeTensor(std::ostream &os,
                     const lite::Scope &scope,
                     const std::string &var_name) {
  // Store all the persistable vars.
  auto *var = scope.FindVar(var_name);
  const auto &tensor = var->Get<lite::Tensor>();
  TensorToStream(os, tensor);
}

/// For navie buffer
template <typename T>
void SetTensorDataNaive(T *out, size_t size, const std::vector<T> &src) {
  CHECK(out);
  CHECK(size == src.size());
  for (size_t i = 0; i < size; ++i) {
    out[i] = src[i];
  }
}

void LoadParamNaive(const std::string &path,
                    lite::Scope *scope,
                    const std::string &name) {
  CHECK(scope);
  auto *tensor = scope->Var(name)->GetMutable<lite::Tensor>();

  // Load param
  naive_buffer::BinaryTable table;
  table.LoadFromFile(path);
  naive_buffer::proto::ParamDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::ParamDesc desc(&pt_desc);

  VLOG(3) << "model version " << desc.ModelVersion();
  CHECK_EQ(desc.TensorVersion(), 0U) << "Only version 0 is supported";

  // Load LoD info
  auto *tgt_lod = tensor->mutable_lod();
  auto desc_lod = desc.LoD();
  tgt_lod->assign(desc_lod.begin(), desc_lod.end());

  // Load Dim info
  tensor->Resize(lite::DDim(desc.Dim()));

  // Load data
  switch (desc.GetDataType()) {
#define DO(data_type__, T)                                               \
  case VarDescAPI::VarDataType::data_type__:                             \
    SetTensorDataNaive<T>(                                               \
        tensor->mutable_data<T>(), tensor->data_size(), desc.Data<T>()); \
    break

    // DO(BOOL, bool);
    DO(FP32, float);
    DO(INT8, int8_t);
    DO(INT16, int16_t);
    DO(INT32, int32_t);
    DO(INT64, int64_t);
#undef DO
    default:
      LOG(FATAL) << "unknown type";
  }
}

void SaveParamNaive(const std::string &path,
                    const lite::Scope &scope,
                    const std::string &var_name) {
  // the 1st field, uint32_t version
  constexpr uint32_t version = 0;

  auto *var = scope.FindVar(var_name);
  const auto &tensor = var->Get<lite::Tensor>();

  naive_buffer::BinaryTable table;
  naive_buffer::proto::ParamDesc pt_desc(&table);
  naive_buffer::ParamDesc desc(&pt_desc);

  desc.SetModelVersion(version);
  desc.SetTensorVersion(version);

  desc.SetLoDLevel(tensor.lod().size());
  desc.SetLoD(tensor.lod());

  // TODO(sangoly): support other data types.
  desc.SetDataType(VarDescAPI::VarDataType::FP32);
  desc.SetDim(tensor.dims().Vectorize());
  uint64_t size = tensor.memory_size();
  CHECK_LT(size, std::numeric_limits<std::streamsize>::max())
      << "Index overflow when writing tensor";

#ifdef LITE_WITH_CUDA
  if (tensor.target() == TARGET(kCUDA)) {
    std::unique_ptr<float> tmp_buffer(new float[tensor.data_size()]);
    TargetWrapperCuda::MemcpySync(tmp_buffer.get(),
                                  tensor.data<float>(),
                                  tensor.data_size(),
                                  IoDirection::DtoH);
    desc.SetData<float>(tmp_buffer.get(), tensor.data_size());
  } else  // NOLINT
#endif    // LITE_WITH_CUDA
  {
    desc.SetData<float>(tensor.data<float>(), tensor.data_size());
  }

  // Save param
  pt_desc.Save();
  table.SaveToFile(path);
}

void LoadModelNaive(const std::string &model_dir,
                    Scope *scope,
                    naive_buffer::proto::ProgramDesc *prog) {
  using block_desc_list_t =
      naive_buffer::ListBuilder<naive_buffer::proto::BlockDesc>;
  using var_desc_list_t =
      naive_buffer::ListBuilder<naive_buffer::proto::VarDesc>;

  CHECK(prog);
  // Load model
  const std::string prog_path = model_dir + "/__model__";
  prog->table()->LoadFromFile(prog_path);
  prog->Load();

  auto *main_block_desc =
      prog->GetMutableField<block_desc_list_t>("blocks")->GetMutable(0);
  auto *var_descs = main_block_desc->GetMutableField<var_desc_list_t>("vars");
  for (size_t i = 0; i < var_descs->size(); ++i) {
    auto *desc = var_descs->GetMutable(i);
    naive_buffer::VarDesc var(desc);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
      continue;

    std::string file_path = model_dir + "/" + var.Name();
    VLOG(4) << "reading weight " << var.Name();

    std::ifstream file(file_path);
    switch (var.GetType()) {
      case VarDescAPI::VarDataType::LOD_TENSOR:
        LoadParamNaive(file_path, scope, var.Name());
        break;
      default:
        CHECK(false) << "unknown weight type";
    }
  }
}

void SaveModelNaive(const std::string &model_dir,
                    Scope *scope,
                    naive_buffer::proto::ProgramDesc *prog) {}

}  // namespace lite
}  // namespace paddle
