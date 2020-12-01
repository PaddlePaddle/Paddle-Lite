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
#include <set>
#include <utility>

#include "lite/api/paddle_api.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/variable.h"
#include "lite/core/version.h"
#include "lite/model_parser/base/apis.h"
#include "lite/model_parser/flatbuffers/io.h"
#include "lite/model_parser/tensor_io.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/model_parser/naive_buffer/combined_params_desc.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"
#endif
#include "lite/utils/io.h"

namespace paddle {
namespace lite {

#ifndef LITE_ON_TINY_PUBLISH
void LoadLoDTensor(model_parser::LoDTensorDeserializer *loader,
                   model_parser::ByteReader *reader,
                   Variable *var) {
  auto *tensor = var->GetMutable<lite::Tensor>();
  CHECK(tensor) << "Can not get allocation of the tensor.";
  CHECK(loader) << "The input argument loader is nullptr.";
  CHECK(var) << "The input argument var is nullptr.";
  loader->LoadWithForwardReader(tensor, reader);
}

std::unique_ptr<framework::proto::ProgramDesc> LoadProgram(
    const std::string &path, const lite_api::CxxModelBuffer &model_buffer) {
  std::unique_ptr<framework::proto::ProgramDesc> main_program(
      new framework::proto::ProgramDesc);
  if (model_buffer.is_empty()) {
    model_parser::BinaryFileReader file(path);
    main_program->ParseFromString(file.ReadForwardToString(file.length()));
  } else {
    main_program->ParseFromString(model_buffer.get_program());
  }
  return main_program;
}

// Load directly to CPU, and latter transfer to other devices.
void LoadParam(const std::string &path, Variable *out) {
  model_parser::BinaryFileReader reader(path);
  model_parser::LoDTensorDeserializer loader;
  LoadLoDTensor(&loader, &reader, out);
}

bool IsPersistable(const cpp::VarDesc &var) {
  if (var.Persistable() && var.GetType() != VarDescAPI::Type::FEED_MINIBATCH &&
      var.GetType() != VarDescAPI::Type::FETCH_LIST &&
      var.GetType() != VarDescAPI::Type::RAW) {
    return true;
  }
  return false;
}

void LoadCombinedParamsPb(const std::string &path,
                          lite::Scope *scope,
                          const cpp::ProgramDesc &cpp_prog,
                          const lite_api::CxxModelBuffer &model_buffer) {
  CHECK(scope) << "The input argument scope is nullptr.";
  auto &prog = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);

  // Get vars
  std::vector<std::string> paramlist;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (!IsPersistable(var)) continue;
    paramlist.push_back(var.Name());
  }
  std::stable_sort(paramlist.begin(), paramlist.end());

  std::unique_ptr<model_parser::ByteReader> reader;
  if (!model_buffer.is_empty()) {
    reader.reset(new model_parser::StringBufferReader(
        const_cast<std::string &&>(model_buffer.get_params())));
  } else {
    reader.reset(new model_parser::BinaryFileReader(path));
  }
  model_parser::LoDTensorDeserializer loader;
  for (size_t i = 0; i < paramlist.size(); ++i) {
    auto *var = scope->Var(paramlist[i]);
    LoadLoDTensor(&loader, reader.get(), var);
  }
  CHECK(reader->ReachEnd()) << "You are not allowed to load partial data via"
                            << " LoadCombinedParamsPb, use LoadParam instead.";
}

void LoadModelPb(const std::string &model_dir,
                 const std::string &model_file,
                 const std::string &param_file,
                 Scope *scope,
                 cpp::ProgramDesc *cpp_prog,
                 bool combined,
                 const lite_api::CxxModelBuffer &model_buffer) {
  CHECK(cpp_prog) << "The input cpp program pointer var is nullptr.";
  CHECK(scope) << "The input scope var is nullptr.";
  cpp_prog->ClearBlocks();

  // Load model
  VLOG(4) << "Start load model program...";
  std::string prog_path = model_dir + "/__model__";
  if (combined) {
    prog_path = model_file;
  }
  framework::proto::ProgramDesc pb_proto_prog =
      *LoadProgram(prog_path, model_buffer);
  pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(pb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  VLOG(4) << "Start load model params...";
  CHECK(!(!combined && !model_buffer.is_empty()))
      << "If you want use the model_from_memory,"
      << " you should load the combined model using cfg.set_model_buffer "
         "interface.";
  if (combined) {
    LoadCombinedParamsPb(param_file, scope, *cpp_prog, model_buffer);
  } else {
    auto main_block = pb_proto_prog.blocks(0);
    for (auto &var : main_block.vars()) {
      if (var.name() == "feed" || var.name() == "fetch" || !var.persistable())
        continue;

      std::string file_path = model_dir + "/" + var.name();
      VLOG(4) << "reading weight " << var.name();

      model_parser::BinaryFileReader reader(file_path);
      model_parser::LoDTensorDeserializer loader;

      switch (var.type().type()) {
        case framework::proto::VarType_Type_LOD_TENSOR:
          LoadLoDTensor(&loader, &reader, scope->Var(var.name()));
          break;
        default:
          CHECK(false) << "unknown weight type";
      }
    }
  }

  VLOG(4) << "Load protobuf model in '" << model_dir << "'' successfully";
}

void SaveModelPb(const std::string &model_dir,
                 const Scope &exec_scope,
                 const cpp::ProgramDesc &cpp_prog,
                 bool combined) {
  MkDirRecur(model_dir);
  // Save program
  framework::proto::ProgramDesc pb_proto_prog;
  pb::ProgramDesc pb_prog(&pb_proto_prog);
  TransformProgramDescCppToAny(cpp_prog, &pb_prog);

  std::string prog_path = model_dir + "/__model__";
  if (combined) {
    prog_path = model_dir + "/model";
  }
  std::ofstream model_ostream(prog_path, std::ios_base::binary);
  CHECK(model_ostream.is_open());
  const std::string pb_str = pb_proto_prog.SerializeAsString();
  model_ostream.write(pb_str.c_str(), pb_str.size());
  model_ostream.close();

  // Save Params
  // NOTE: Only main block be used now.
  if (combined) {
    const std::string combined_params_path = model_dir + "/params";
    SaveCombinedParamsPb(combined_params_path, exec_scope, cpp_prog);
  } else {
    for (auto &item : pb_proto_prog.blocks(0).vars()) {
      if (item.name() == "feed" || item.name() == "fetch" ||
          !item.persistable())
        continue;
      const std::string path = model_dir + "/" + item.name();

      model_parser::BinaryFileWriter file(path);
      model_parser::LoDTensorSerializer saver;
      auto *var = exec_scope.FindVar(item.name());
      const auto &tensor = var->Get<lite::Tensor>();
      if (tensor.target() == TARGET(kCUDA)) {
        LOG(FATAL) << "The storage of the device Tensor is to be implemented, "
                      "please copy it to the Host Tensor temporarily.";
      }
      saver.SaveWithForwardWriter(tensor, &file);
    }
  }
  VLOG(4) << "Save protobuf model in '" << model_dir << "'' successfully";
}

void SaveCombinedParamsPb(const std::string &path,
                          const lite::Scope &exec_scope,
                          const cpp::ProgramDesc &cpp_prog) {
  auto &prog = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);

  // Get vars
  std::vector<std::string> paramlist;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (!IsPersistable(var)) continue;
    paramlist.push_back(var.Name());
  }
  std::stable_sort(paramlist.begin(), paramlist.end());

  // Save vars
  model_parser::BinaryFileWriter file(path);
  model_parser::LoDTensorSerializer saver;
  for (size_t i = 0; i < paramlist.size(); ++i) {
    auto *var = exec_scope.FindVar(paramlist[i]);
    const auto &tensor = var->Get<lite::Tensor>();
    if (tensor.target() == TARGET(kCUDA)) {
      LOG(FATAL) << "The storage of the device Tensor is to be implemented, "
                    "please copy it to the Host Tensor temporarily.";
    }
    saver.SaveWithForwardWriter(tensor, &file);
  }
}

/// For navie buffer
void SetParamInfoNaive(naive_buffer::ParamDesc *param_desc,
                       const lite::Scope &scope,
                       const std::string &var_name) {
  CHECK(param_desc);
  auto &desc = *param_desc;

  // the 1st field, uint32_t version
  constexpr uint32_t version = 0;

  auto *var = scope.FindVar(var_name);
  const auto &tensor = var->Get<lite::Tensor>();

  desc.SetName(var_name);

  desc.SetModelVersion(version);
  desc.SetTensorVersion(version);

  desc.SetLoDLevel(tensor.lod().size());
  desc.SetLoD(tensor.lod());

  // TODO(sangoly): support other data types.
  switch (tensor.precision()) {
#define SET_DATA_TYPE(precision, type_desc) \
  case precision:                           \
    desc.SetDataType(type_desc);            \
    break;

    SET_DATA_TYPE(PRECISION(kFloat), VarDescAPI::VarDataType::FP32);
    SET_DATA_TYPE(PRECISION(kInt8), VarDescAPI::VarDataType::INT8);
    SET_DATA_TYPE(PRECISION(kInt16), VarDescAPI::VarDataType::INT16);
    SET_DATA_TYPE(PRECISION(kInt32), VarDescAPI::VarDataType::INT32);
    SET_DATA_TYPE(PRECISION(kInt64), VarDescAPI::VarDataType::INT64);
#undef SET_DATA_TYPE
    default:
      LOG(FATAL) << "unknown precision type: "
                 << PrecisionToStr(tensor.precision());
  }
  desc.SetDim(tensor.dims().Vectorize());
  uint64_t size = tensor.memory_size();
  CHECK_LT(size, (std::numeric_limits<std::streamsize>::max)())
      << "Index overflow when writing tensor";

#ifdef LITE_WITH_CUDA
  if (tensor.target() == TARGET(kCUDA)) {
    switch (tensor.precision()) {
#define DO(precision, type)                                         \
  case precision: {                                                 \
    std::unique_ptr<type> tmp_buffer(new type[tensor.data_size()]); \
    TargetWrapperCuda::MemcpySync(tmp_buffer.get(),                 \
                                  tensor.data<type>(),              \
                                  tensor.data_size(),               \
                                  IoDirection::DtoH);               \
    desc.SetData<type>(tmp_buffer.get(), tensor.data_size());       \
  } break;
      DO(PRECISION(kFloat), float);
      DO(PRECISION(kInt8), int8_t);
      DO(PRECISION(kInt16), int16_t);
      DO(PRECISION(kInt32), int32_t);
      DO(PRECISION(kInt64), int64_t);
#undef DO
      default:
        LOG(FATAL) << "unknown precision type: "
                   << PrecisionToStr(tensor.precision());
    }
  } else  // NOLINT
#endif    // LITE_WITH_CUDA
  {
    switch (tensor.precision()) {
#define DO(precision, type)                                      \
  case precision:                                                \
    desc.SetData<type>(tensor.data<type>(), tensor.data_size()); \
    break;
      DO(PRECISION(kFloat), float);
      DO(PRECISION(kInt8), int8_t);
      DO(PRECISION(kInt16), int16_t);
      DO(PRECISION(kInt32), int32_t);
      DO(PRECISION(kInt64), int64_t);
#undef DO
      default:
        LOG(FATAL) << "unknown precision type: "
                   << PrecisionToStr(tensor.precision());
    }
  }
}

void SaveParamNaive(const std::string &path,
                    const lite::Scope &scope,
                    const std::string &var_name) {
  naive_buffer::BinaryTable table;
  naive_buffer::proto::ParamDesc pt_desc(&table);
  naive_buffer::ParamDesc desc(&pt_desc);

  SetParamInfoNaive(&desc, scope, var_name);

  // Save param
  pt_desc.Save();
  table.SaveToFile(path);
}

void SaveCombinedParamsNaive(const std::string &path,
                             const lite::Scope &exec_scope,
                             const cpp::ProgramDesc &cpp_prog) {
  naive_buffer::BinaryTable table;
  naive_buffer::proto::CombinedParamsDesc pt_desc(&table);
  naive_buffer::CombinedParamsDesc desc(&pt_desc);

  auto &prog = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);
  // set unique_var_names to avoid saving shared params repeatedly
  std::set<std::string> unique_var_names;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable() ||
        unique_var_names.count(var.Name()) > 0)
      continue;
    naive_buffer::ParamDesc param_desc(desc.AddParam());
    SetParamInfoNaive(&param_desc, exec_scope, var.Name());
    unique_var_names.emplace(var.Name());
  }

  pt_desc.Save();
  table.AppendToFile(path);
}

////////////////////////////////////////////////////////////////////////////////////
// Save model: meta_version = 1
// Flatbuffer model + params
////////////////////////////////////////////////////////////////////////////////////
// Create a new file and write data into it.
void WriteToFile(const std::string &filename,
                 const void *src,
                 size_t byte_size) {
  CHECK(src);
  FILE *file = fopen(filename.c_str(), "wb");
  CHECK(file);
  CHECK(fwrite(src, sizeof(char), byte_size, file) == byte_size);
  fclose(file);
}
// Append data into an existed file.
void AppendToFile(const std::string &filename,
                  const void *src,
                  size_t byte_size) {
  CHECK(src);
  FILE *fp = fopen(filename.c_str(), "ab");
  CHECK(fp) << "Unable to open file: " << filename;
  if (fwrite(reinterpret_cast<const char *>(src), 1, byte_size, fp) !=
      byte_size) {
    fclose(fp);
    LOG(FATAL) << "Write file error: " << filename;
  }
  fclose(fp);
}
/* ---------- Flatbuffers ---------- */
void SaveModelNaive(const std::string &model_file,
                    const Scope &exec_scope,
                    const cpp::ProgramDesc &cpp_prog) {
  /* 1. Save model to model.fbs */
  const std::string prog_path = model_file + ".nb";
  // Save meta_version(uint16) into file
  uint16_t meta_version = 1;
  WriteToFile(prog_path, &meta_version, sizeof(uint16_t));

  // Save lite_version(char[16]) into file
  const int paddle_version_length = 16 * sizeof(char);
  std::string paddle_version = version();
  AppendToFile(prog_path, paddle_version.c_str(), paddle_version_length);
  VLOG(4) << "paddle_version:" << paddle_version;

  fbs::ProgramDesc fbs_prog;
  TransformProgramDescCppToAny(cpp_prog, &fbs_prog);
  uint64_t topology_size = (fbs_prog.data()).size();
  AppendToFile(prog_path, &topology_size, sizeof(uint64_t));
  /* 1. Save model to model.fbs */
  AppendToFile(prog_path, (fbs_prog.data()).data(), topology_size);
  VLOG(4) << "save topology_size:" << topology_size;

  /* 2. Get param names from cpp::ProgramDesc */
  auto &main_block_desc = *cpp_prog.GetBlock<cpp::BlockDesc>(0);
  // set unique_var_names to avoid saving shared params repeatedly
  std::set<std::string> unique_var_names;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable() ||
        unique_var_names.count(var.Name()) > 0)
      continue;
    unique_var_names.emplace(var.Name());
  }

  /* 3. Save combined params to params.fbs */
  fbs::CombinedParamsDesc params_prog;
  fbs::SetCombinedParamsWithScope(exec_scope, unique_var_names, &params_prog);
  auto data_cache = params_prog.data();
  AppendToFile(prog_path, data_cache.data(), data_cache.size());

  LOG(INFO) << "Save naive buffer model in '" << prog_path << " successfully";
}

template <typename T>
void SetTensorDataNaive(T *out, size_t size, const std::vector<T> &src) {
  CHECK(out);
  CHECK(size == src.size());
  for (size_t i = 0; i < size; ++i) {
    out[i] = src[i];
  }
}

void GetParamInfoNaive(const naive_buffer::ParamDesc &desc,
                       lite::Scope *scope,
                       const std::string &name) {
  CHECK(scope);
  CHECK_EQ(desc.Name(), name)
      << "Var name not equal: ParamDesc.name=" << desc.Name()
      << "vs filename=" << name;

  auto *tensor = scope->Var(name)->GetMutable<lite::Tensor>();

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
#define SET_TENSOR(data_type__, T, precision)                            \
  case VarDescAPI::VarDataType::data_type__:                             \
    SetTensorDataNaive<T>(                                               \
        tensor->mutable_data<T>(), tensor->data_size(), desc.Data<T>()); \
    tensor->set_precision(precision);                                    \
    break

    // SET_TENSOR(BOOL, bool, PRECISION(kBool));
    SET_TENSOR(FP64, double, PRECISION(kFP64));
    SET_TENSOR(FP32, float, PRECISION(kFloat));
    SET_TENSOR(UINT8, uint8_t, PRECISION(kUInt8));
    SET_TENSOR(INT8, int8_t, PRECISION(kInt8));
    SET_TENSOR(INT16, int16_t, PRECISION(kInt16));
    SET_TENSOR(INT32, int32_t, PRECISION(kInt32));
    SET_TENSOR(INT64, int64_t, PRECISION(kInt64));
#undef SET_TENSOR
    default:
      LOG(FATAL) << "unknown type";
  }
  tensor->set_persistable(true);
}

void LoadParamNaive(const std::string &path,
                    lite::Scope *scope,
                    const std::string &name) {
  // Load param
  naive_buffer::BinaryTable table;
  table.LoadFromFile(path);
  naive_buffer::proto::ParamDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::ParamDesc desc(&pt_desc);
  GetParamInfoNaive(desc, scope, name);
}

void LoadCombinedParamsNaive(const std::string &path,
                             const uint64_t &offset,
                             lite::Scope *scope,
                             const cpp::ProgramDesc &cpp_prog,
                             bool params_from_memory) {
  naive_buffer::BinaryTable table;
  if (params_from_memory) {
    table.LoadFromMemory(path.c_str() + offset, path.length() - offset);
  } else {
    table.LoadFromFile(path, offset, 0);
  }
  naive_buffer::proto::CombinedParamsDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::CombinedParamsDesc desc(&pt_desc);

  std::set<std::string> param_names;
  for (size_t i = 0; i < desc.ParamsSize(); ++i) {
    naive_buffer::ParamDesc param_desc(desc.GetParam(i));
    GetParamInfoNaive(param_desc, scope, param_desc.Name());
    param_names.insert(param_desc.Name());
  }

  // Check all params loaded
  auto &prog = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
      continue;
    CHECK(param_names.count(var.Name())) << "Persistable var[" << var.Name()
                                         << "] not found";
  }
}

///////////////////////////////////////////////////////////////////////////////
/* Old Method of loading and saving model, before V2.3.0                     */
/* Warning: this is an old inference and will be abandened in release/v3.0.0 */
///////////////////////////////////////////////////////////////////////////////
void LoadModelNaive(const std::string &model_dir,
                    Scope *scope,
                    cpp::ProgramDesc *cpp_prog,
                    bool combined) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  LOG(WARNING)
      << "WARNING: MobileConfig::set_model_dir and "
         "MobileConfig::set_model_buffer are deprecated APIs "
         "and will be removed in latter release. \n"
         "    MobileConfig::set_model_from_file(const std::string& model_file)"
         " and MobileConfig::set_model_from_buffer(const std::string& "
         "model_buffer) are recommended.";
  // Load model
  const std::string prog_path = model_dir + "/__model__.nb";
  naive_buffer::BinaryTable table;
  table.LoadFromFile(prog_path);
  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  if (combined) {
    const std::string combined_params_path = model_dir + "/param.nb";
    LoadCombinedParamsNaive(combined_params_path, 0, scope, *cpp_prog, false);
  } else {
    auto &prog = *cpp_prog;
    auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);
    for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
      auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
      if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
        continue;

      std::string file_path = model_dir + "/" + var.Name() + ".nb";
      VLOG(4) << "reading weight " << var.Name();

      switch (var.GetType()) {
        case VarDescAPI::Type::LOD_TENSOR:
          LoadParamNaive(file_path, scope, var.Name());
          break;
        default:
          CHECK(false) << "unknown weight type";
      }
    }
  }

  VLOG(4) << "Load naive buffer model in '" << model_dir << "' successfully";
}

void LoadModelNaiveFromMemory(const std::string &model_buffer,
                              const std::string &param_buffer,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  // Load model
  naive_buffer::BinaryTable table;
  table.LoadFromMemory(model_buffer.c_str(), model_buffer.length());

  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  LoadCombinedParamsNaive(param_buffer, 0, scope, *cpp_prog, true);

  VLOG(4) << "Load model from naive buffer memory successfully";
}
#endif  // LITE_ON_TINY_PUBLISH
//////////////////////////////////////////////////////////////////////

// usage: LoadModelNaiveFromFile is used for loading model from file.
template <typename T>
void ReadModelDataFromFile(T *data,
                           const std::string &prog_path,
                           uint64_t *offset,
                           const uint64_t &size) {
  model_parser::Buffer prog_data =
      lite::fbs::LoadFile(prog_path, *offset, size);
  memcpy(data, prog_data.data(), size);
  *offset = *offset + size;
}
/*
 * Binary structure of naive_buffer model: model.nb
 * ----------------------------------------------------------
 * |       |    PART         |   Precision |   Length(byte) |
 * |   1   |  meta_version   |   uint16_t  |       2        |
 * |   2   |  opt_version    |   char[16]  |      16        |
 * |   3   |  topo_size      |   uint64_t  |       8        |
 * |   4   |  topo_data      |   char[]    | topo_size byte |
 * |   5   |  param_data     |   char[]    |                |
 * ----------------------------------------------------------
 *  Meaning of each part:
 *      meta_version: meata_version, 0 default.
 *      opt_version:  lite_version of opt tool that transformed this model.
 *      topo_size:    length of `topo_data`.
 *      topo_data:    contains model's topology data.
 *      param_data:   contains model's params data.
*/

void LoadModelNaiveFromFile(const std::string &filename,
                            Scope *scope,
                            cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  // ModelFile
  const std::string prog_path = filename;

  // Offset
  uint64_t offset = 0;

  // (1)get meta version
  uint16_t meta_version;
  ReadModelDataFromFile<uint16_t>(
      &meta_version, prog_path, &offset, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;

  switch (meta_version) {
    case 0:
#ifndef LITE_ON_TINY_PUBLISH
      LoadModelNaiveV0FromFile(filename, scope, cpp_prog);
#else
      LOG(FATAL) << "Paddle-Lite v2.7 has upgraded the naive-buffer model "
                    "format. Please use the OPT to generate a new model. "
                    "Thanks!";
#endif
      break;
    case 1:
      LoadModelFbsFromFile(filename, scope, cpp_prog);
      break;
    default:
      LOG(FATAL) << "The model format cannot be recognized. Please make sure "
                    "you use the correct interface and model file.";
      break;
  }
}
#ifndef LITE_ON_TINY_PUBLISH
void LoadModelNaiveV0FromFile(const std::string &filename,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();
  // ModelFile
  const std::string prog_path = filename;

  // Offset
  uint64_t offset = 0;

  // (1)get meta version
  uint16_t meta_version;
  ReadModelDataFromFile<uint16_t>(
      &meta_version, prog_path, &offset, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;

  // (2)get opt version
  char opt_version[16];
  const uint64_t opt_version_length = 16 * sizeof(char);
  ReadModelDataFromFile<char>(
      opt_version, prog_path, &offset, opt_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);

  // check version, opt's version should be consistent with current Paddle-Lite
  // version.
  const std::string paddle_version = version();
  const std::string opt_version_str = opt_version;
  if (paddle_version != opt_version_str) {
    LOG(WARNING) << "warning: the version of opt that transformed this model "
                    "is not consistent with current Paddle-Lite version."
                    "\n      version of opt:"
                 << static_cast<const char *>(opt_version)
                 << "\n      version of current Paddle-Lite:" << paddle_version;
  }

  // (3)get topo_size
  uint64_t topo_size;
  ReadModelDataFromFile<uint64_t>(
      &topo_size, prog_path, &offset, sizeof(uint64_t));

  // (4)get topo data
  naive_buffer::BinaryTable topo_table;
  topo_table.LoadFromFile(prog_path, offset, topo_size);
  offset = offset + topo_size;
  // transform topo_data into cpp::ProgramDesc
  naive_buffer::proto::ProgramDesc nb_proto_prog(&topo_table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // (5)Load Params
  LoadCombinedParamsNaive(prog_path, offset, scope, *cpp_prog, false);

  VLOG(4) << "Load naive buffer model in '" << filename << "' successfully";
}
#endif  // LITE_ON_TINY_PUBLISH
void LoadModelFbsFromFile(const std::string &filename,
                          Scope *scope,
                          cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  CHECK_EQ(cpp_prog->BlocksSize(), 0);
  // Offset
  uint64_t offset = sizeof(uint16_t);

  // get opt version
  char opt_version[16];
  const uint64_t opt_version_length = 16 * sizeof(char);
  ReadModelDataFromFile<char>(
      opt_version, filename, &offset, opt_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);
  // check version, opt's version should be consistent with current Paddle-Lite
  // version.
  const std::string paddle_version = version();
  const std::string opt_version_str = opt_version;
  if (paddle_version != opt_version_str) {
    LOG(WARNING) << "warning: the version of opt that transformed this model "
                    "is not consistent with current Paddle-Lite version."
                    "\n      version of opt:"
                 << static_cast<const char *>(opt_version)
                 << "\n      version of current Paddle-Lite:" << paddle_version;
  }
  // (3)get topo_size
  uint64_t topo_size;
  ReadModelDataFromFile<uint64_t>(
      &topo_size, filename, &offset, sizeof(uint64_t));

#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  cpp_prog->Init(fbs::LoadFile(filename, offset, topo_size));
#elif LITE_ON_TINY_PUBLISH
  LOG(FATAL) << "Since no data structure of Flatbuffers has been constructed, "
                "the model cannot be loaded.";
#else
  fbs::ProgramDesc program(fbs::LoadFile(filename, offset, topo_size));
  TransformProgramDescAnyToCpp(program, cpp_prog);
#endif
  offset = offset + topo_size;

  /* 2. Load scope from params.fbs */
  fbs::CombinedParamsDescView params(fbs::LoadFile(filename, offset));
  fbs::SetScopeWithCombinedParams(scope, params);

  VLOG(4) << "Load naive buffer model in '" << filename << "' successfully";
}

// usage: LoadModelNaiveFromMemory is used for loading naive model from memory
template <typename T>
void ReadModelDataFromBuffer(T *data,
                             const std::string &model_buffer,
                             uint64_t *offset,
                             const uint64_t &size) {
  memcpy(data, model_buffer.c_str() + *offset, size);
  *offset = *offset + size;
}

void LoadModelNaiveFromMemory(const std::string &model_buffer,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  uint64_t offset = 0;
  // (1)get meta version
  uint16_t meta_version;
  ReadModelDataFromBuffer<uint16_t>(
      &meta_version, model_buffer, &offset, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;
  switch (meta_version) {
    case 0:
#ifndef LITE_ON_TINY_PUBLISH
      LoadModelNaiveV0FromMemory(model_buffer, scope, cpp_prog);
#else
      LOG(FATAL) << "Paddle-Lite v2.7 has upgraded the naive-buffer model "
                    "format. Please use the OPT to generate a new model. "
                    "Thanks!";
#endif
      break;
    case 1:
      LoadModelNaiveV1FromMemory(model_buffer, scope, cpp_prog);
      break;
    default:
      LOG(FATAL) << "The model format cannot be recognized. Please make sure "
                    "you use the correct interface and model file.";
      break;
  }
}
#ifndef LITE_ON_TINY_PUBLISH
void LoadModelNaiveV0FromMemory(const std::string &model_buffer,
                                Scope *scope,
                                cpp::ProgramDesc *cpp_prog) {
  // Offset
  uint64_t offset = sizeof(uint16_t);

  // (2)get opt version
  char opt_version[16];
  const uint64_t paddle_version_length = 16 * sizeof(char);
  ReadModelDataFromBuffer<char>(
      opt_version, model_buffer, &offset, paddle_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);

  // (3)get topo_size and topo_data
  uint64_t topo_size;
  ReadModelDataFromBuffer<uint64_t>(
      &topo_size, model_buffer, &offset, sizeof(uint64_t));
  naive_buffer::BinaryTable table;
  table.LoadFromMemory(model_buffer.c_str() + offset, topo_size);
  offset = offset + topo_size;

  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  // only combined Params are supported in Loading Model from memory
  LoadCombinedParamsNaive(model_buffer, offset, scope, *cpp_prog, true);

  VLOG(4) << "Load model from naive buffer memory successfully";
}
#endif
///////////////////////////////////////////////////////////////////
// Meta_version=1
///////////////////////////////////////////////////////////////////
void LoadModelNaiveV1FromMemory(const std::string &model_buffer,
                                Scope *scope,
                                cpp::ProgramDesc *cpp_prog) {
  // Offset
  uint64_t offset = sizeof(uint16_t);

  // (2)get opt version
  char opt_version[16];
  const uint64_t paddle_version_length = 16 * sizeof(char);
  ReadModelDataFromBuffer<char>(
      opt_version, model_buffer, &offset, paddle_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);

  // (3)get prog_size and prog_data
  uint64_t prog_size;
  ReadModelDataFromBuffer<uint64_t>(
      &prog_size, model_buffer, &offset, sizeof(uint64_t));
  VLOG(4) << "prog_size:" << prog_size;

  model_parser::Buffer prog_data(prog_size);
  memcpy(prog_data.data(), model_buffer.c_str() + offset, prog_size);
#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  cpp_prog->Init(std::move(prog_data));
#elif LITE_ON_TINY_PUBLISH
  LOG(FATAL) << "Since no data structure of Flatbuffers has been constructed, "
                "the model cannot be loaded.";
#else
  fbs::ProgramDesc program(prog_data);
  TransformProgramDescAnyToCpp(program, cpp_prog);
#endif
  offset = offset + prog_size;
  VLOG(4) << "param_size:" << model_buffer.length() - offset;

  model_parser::Buffer params_data(model_buffer.length() - offset);
  memcpy(params_data.data(),
         model_buffer.c_str() + offset,
         model_buffer.length() - offset);

  fbs::CombinedParamsDescView params(std::move(params_data));
  fbs::SetScopeWithCombinedParams(scope, params);

  VLOG(4) << "Load model from naive buffer memory successfully";
}

}  // namespace lite
}  // namespace paddle
