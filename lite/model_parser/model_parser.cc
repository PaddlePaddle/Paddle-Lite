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
#include "lite/model_parser/pb/tensor_io.h"
#ifndef LITE_ON_TINY_PUBLISH
#include <cstdio>
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"
#endif
#include "lite/utils/io.h"
namespace paddle {
namespace lite {
#ifndef LITE_ON_TINY_PUBLISH
void LoadLoDTensor(model_parser::pb::LoDTensorDeserializer *loader,
                   model_parser::ByteReader *reader,
                   Variable *var) {
  auto *tensor = var->GetMutable<lite::Tensor>();
  CHECK(tensor) << "Can not get allocation of the tensor.";
  CHECK(loader) << "The input argument loader is nullptr.";
  CHECK(var) << "The input argument var is nullptr.";
  loader->ForwardRead(tensor, reader);
}

std::unique_ptr<framework::proto::ProgramDesc> LoadProgram(
    const std::string &path, const lite_api::CxxModelBuffer &model_buffer) {
  std::unique_ptr<framework::proto::ProgramDesc> main_program(
      new framework::proto::ProgramDesc);
  if (model_buffer.is_empty()) {
    model_parser::BinaryFileReader file(path);
    main_program->ParseFromString(file.ReadToString(file.length()));
  } else {
    main_program->ParseFromString(model_buffer.get_program());
  }
  return main_program;
}

// Load directly to CPU, and latter transfer to other devices.
void LoadParam(const std::string &path, Variable *out) {
  model_parser::BinaryFileReader reader(path);
  model_parser::pb::LoDTensorDeserializer loader;
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
    reader.reset(
        new model_parser::StringBufferReader(model_buffer.get_params()));
  } else {
    reader.reset(new model_parser::BinaryFileReader(path));
  }
  model_parser::pb::LoDTensorDeserializer loader;
  if (!paramlist.empty()) {
    CHECK(reader->length())
        << "The model needs weights but the weight file is not existed.";
  }
  for (size_t i = 0; i < paramlist.size(); ++i) {
    auto *var = scope->Var(paramlist[i]);
    LoadLoDTensor(&loader, reader.get(), var);
  }
  CHECK(reader->ReachEnd()) << "You are not allowed to load partial data via"
                            << " LoadCombinedParamsPb, use LoadParam instead.";
}

// Print error message about LoadModelPb
void PrintPbModelErrorMessage() {
  LOG(FATAL) << "\n Error, Unsupported model format!\n"
             << "      1. contents in model directory should be in one of "
                "these formats:\n"
             << "          (1) __model__ + var1 + var2 + etc.\n"
             << "          (2) model + var1 + var2 + etc.\n"
             << "          (3) model.pdmodel + model.pdiparams\n"
             << "          (4) model + params\n"
             << "          (5) model + weights\n"
             << "      2. You can also appoint the model and params file in "
                "custom format:\n"
             << "          eg. |-- set_model_file('custom_model_name')\n"
             << "              |-- set_param_file('custom_params_name')'";
}
// Find correct model filename
std::string FindModelFileName(const std::string &model_dir,
                              const std::string &model_file,
                              bool combined) {
  std::string prog_path;
  if (!combined) {
    // format 1. model_dir/__model__
    // format 2. model_dir/model
    // format 3. model_dir/pdmodel
    if (IsFileExists(model_dir + "/__model__")) {
      prog_path = model_dir + "/__model__";
    } else if (IsFileExists(model_dir + "/model")) {
      prog_path = model_dir + "/model";
    } else if (IsFileExists(model_dir + "/model.pdmodel")) {
      prog_path = model_dir + "/model.pdmodel";
    } else {
      PrintPbModelErrorMessage();
    }
  } else {
    if (IsFileExists(model_file)) {
      prog_path = model_file;
    } else {
      LOG(FATAL) << "\nError, the model file '" << model_file
                 << "' is not existed. Please confirm that you have inputed "
                    "correct model file path.";
    }
  }
  return prog_path;
}

// load noncombined params from directory.
void LoadNonCombinedParamsPb(const std::string &model_dir,
                             cpp::ProgramDesc *cpp_prog,
                             const lite_api::CxxModelBuffer &model_buffer,
                             Scope *scope) {
  auto *main_block = cpp_prog->GetBlock<cpp::BlockDesc>(0);
  std::string log_info = "Loading non-combined params data from " + model_dir;
  // Check param files format
  // default format: non-combined params
  for (auto &var : main_block->GetVars()) {
    if (IsParamVarDesc(*var)) {
      if (IsFileExists(model_dir + "/" + var->Name())) {
        VLOG(4) << "reading weight " << var->Name();
        model_parser::BinaryFileReader reader(model_dir + "/" + var->Name());
        model_parser::pb::LoDTensorDeserializer loader;
        switch (var->GetType()) {
          case VarDescAPI::Type::LOD_TENSOR:
            LoadLoDTensor(&loader, &reader, scope->Var(var->Name()));
            break;
          default:
            CHECK(false) << "unknown weight type";
        }
      } else {
        std::string params_path{""};
        // format 1. model_dir/params
        // format 2. model_dir/weights
        // format 3. model_dir/pdiparams
        if (IsFileExists(model_dir + "/params")) {
          params_path = model_dir + "/params";
        } else if (IsFileExists(model_dir + "/weights")) {
          params_path = model_dir + "/weights";
        } else if (IsFileExists(model_dir + "/model.pdiparams")) {
          params_path = model_dir + "/model.pdiparams";
        } else {
          PrintPbModelErrorMessage();
        }
        log_info = "Loading params data from " + params_path;
        LoadCombinedParamsPb(params_path, scope, *cpp_prog, model_buffer);
        break;
      }
    }
  }
  OPT_LOG << log_info;
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

  // Load model topology data from file.
  std::string prog_path =
      model_buffer.is_empty()
          ? FindModelFileName(model_dir, model_file, combined)
          : "";
  OPT_LOG << "Loading topology data from " << prog_path;
  framework::proto::ProgramDesc pb_proto_prog =
      *LoadProgram(prog_path, model_buffer);
  pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(pb_prog, cpp_prog);

  // Load params data from file.
  // NOTE: Only main block be used now.
  CHECK(combined || model_buffer.is_empty())
      << "If you want use the model_from_memory,"
      << " you should load the combined model using cfg.set_model_buffer "
         "interface.";
  if (!combined) {
    LoadNonCombinedParamsPb(model_dir, cpp_prog, model_buffer, scope);
  } else {
    if (model_buffer.is_empty()) {
      OPT_LOG << "Loading params data from " << param_file;
      CHECK(IsFileExists(param_file))
          << "Error, the param file '" << param_file
          << "' is not existed. Please confirm that you have inputed "
             "correct param file path.";
    }

    LoadCombinedParamsPb(param_file, scope, *cpp_prog, model_buffer);
  }
  OPT_LOG << "1. Model is successfully loaded!";
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
      if (!pb::IsParamVarDesc(item)) continue;
      const std::string path = model_dir + "/" + item.name();

      model_parser::BinaryFileWriter file(path);
      model_parser::pb::LoDTensorSerializer saver;
      auto *var = exec_scope.FindVar(item.name());
      const auto &tensor = var->Get<lite::Tensor>();
      if (tensor.target() == TARGET(kCUDA)) {
        LOG(FATAL) << "The storage of the device Tensor is to be implemented, "
                      "please copy it to the Host Tensor temporarily.";
      }
      saver.ForwardWrite(tensor, &file);
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
  model_parser::pb::LoDTensorSerializer saver;
  for (size_t i = 0; i < paramlist.size(); ++i) {
    auto *var = exec_scope.FindVar(paramlist[i]);
    const auto &tensor = var->Get<lite::Tensor>();
    if (tensor.target() == TARGET(kCUDA)) {
      LOG(FATAL) << "The storage of the device Tensor is to be implemented, "
                    "please copy it to the Host Tensor temporarily.";
    }
    saver.ForwardWrite(tensor, &file);
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Save model: meta_version = 1
// Flatbuffer model + params
////////////////////////////////////////////////////////////////////////////////////
/* ---------- Flatbuffers ---------- */
void SaveModelNaive(const std::string &model_file,
                    const Scope &exec_scope,
                    const cpp::ProgramDesc &cpp_prog) {
  model_parser::Buffer buffer;
  /* 1. Save model to model.fbs */
  const std::string prog_path = model_file + ".nb";
  model_parser::BinaryFileWriter writer{prog_path};

  // Meta_version(uint16), default value is 2.
  uint16_t meta_version = 2;
  // You can modify meta_version by register environment variable
  // 'PADDLE_LITE_MODEL_VERSION1'
  const char *PADDLE_LITE_EXPERIMENTAL_MODEL =
      std::getenv("PADDLE_LITE_MODEL_VERSION1");
  if (PADDLE_LITE_EXPERIMENTAL_MODEL != nullptr) {
    meta_version = 1;
  }
  // Save meta_version(uint16) into file
  writer.Write(&meta_version, sizeof(uint16_t));

  // Save lite_version(char[16]) into file
  const int paddle_version_length = 16 * sizeof(char);
  std::string paddle_version = version();
  writer.Write(paddle_version.c_str(), paddle_version_length);
  VLOG(4) << "paddle_version:" << paddle_version;

  /* 1. Get topolygy description from cpp::ProgramDesc */
  fbs::ProgramDesc fbs_prog;
  TransformProgramDescCppToAny(cpp_prog, &fbs_prog);
  fbs_prog.CopyDataToBuffer(&buffer);
  uint64_t topology_size = buffer.size();
  // Save topolygy description into naive model
  writer.Write(&topology_size, sizeof(uint64_t));
  writer.Write(buffer.data(), topology_size);
  VLOG(4) << "save topology_size:" << topology_size;

  /* 2. Get param names from cpp::ProgramDesc */
  auto &main_block_desc = *cpp_prog.GetBlock<cpp::BlockDesc>(0);
  // set unique_var_names to avoid saving shared params repeatedly
  std::set<std::string> unique_var_names;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (!IsParamVarDesc(var) || unique_var_names.count(var.Name()) > 0)
      continue;
    unique_var_names.emplace(var.Name());
  }

  /* 3. Save paramdesc info into model file */
  switch (meta_version) {
    case 1: {
      /* 3.1 Save combined params to params.fbs */
      fbs::CombinedParamsDesc params_prog;
      fbs::deprecated::SetCombinedParamsWithScope(
          exec_scope, unique_var_names, &params_prog);
      params_prog.CopyDataToBuffer(&buffer);
      writer.Write(buffer.data(), buffer.size());
      break;
    }
    case 2: {
      fbs::ParamSerializer serializer{&writer};
      // 3.2 Save params into naive model
      serializer.ForwardWrite(exec_scope, unique_var_names);
      break;
    }
    default: {
      LOG(FATAL) << "Error: Unsupported opt meta_version, "
                    "meta_version should be set as 1 or 2.";
      break;
    }
  }
  OPT_LOG << "2. Model is optimized and saved into " << prog_path
          << " successfully";
}
#endif  // LITE_ON_TINY_PUBLISH

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
  model_parser::BinaryFileReader reader(filename, 0);

  // (1)get meta version
  uint16_t meta_version;
  reader.Read(&meta_version, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;

  switch (meta_version) {
    case 0:
      LOG(FATAL) << "Paddle-Lite v2.7 has upgraded the naive-buffer model "
                    "format. Please use the OPT to generate a new model. "
                    "Thanks!";
      break;
    case 1:
      LoadModelFbsFromFile(&reader, scope, cpp_prog, 1);
      break;
    case 2:
      LoadModelFbsFromFile(&reader, scope, cpp_prog, 2);
      break;
    default:
      LOG(FATAL) << "The model format cannot be recognized. Please make sure "
                    "you use the correct interface and model file.";
      break;
  }
  VLOG(4) << "Load naive buffer model in '" << filename << "' successfully";
}

void LoadModelFbsFromFile(model_parser::BinaryFileReader *reader,
                          Scope *scope,
                          cpp::ProgramDesc *cpp_prog,
                          uint16_t meta_version) {
  CHECK(cpp_prog);
  CHECK(scope);
  CHECK_EQ(cpp_prog->BlocksSize(), 0);

  // get opt version
  char opt_version[16];
  const uint64_t opt_version_length = 16 * sizeof(char);
  reader->Read(opt_version, opt_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);
  // check version, opt's version should be consistent with current Paddle-Lite
  // version.
  const std::string paddle_version = version();
  const std::string opt_version_str = opt_version;
  if (paddle_version != opt_version_str) {
    LOG(WARNING) << "\nwarning: the version of opt that transformed this model "
                    "is not consistent with current Paddle-Lite version."
                    "\n      version of opt:"
                 << static_cast<const char *>(opt_version)
                 << "\n      version of current Paddle-Lite:" << paddle_version;
  }
  // (3)get topo_size
  uint64_t topo_size;
  reader->Read(&topo_size, sizeof(uint64_t));
  VLOG(4) << "topo_size: " << topo_size;

#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  lite::model_parser::Buffer buf(topo_size);
  reader->Read(buf.data(), topo_size);
  cpp_prog->Init(std::move(buf));
#elif LITE_ON_TINY_PUBLISH
  LOG(FATAL) << "Since no data structure of Flatbuffers has been constructed, "
                "the model cannot be loaded.";
#else
  lite::model_parser::Buffer buf(topo_size);
  reader->Read(buf.data(), topo_size);
  fbs::ProgramDesc program(buf);
  TransformProgramDescAnyToCpp(program, cpp_prog);
#endif

  /* 2. Load scope from params.fbs */
  switch (meta_version) {
    case 1: {
      /* load scope from param.fbs with meta_version=1 */
      lite::model_parser::Buffer buf(reader->length() - reader->current());
      reader->Read(buf.data(), reader->length() - reader->current());
      fbs::CombinedParamsDescView params(std::move(buf));
      fbs::deprecated::SetScopeWithCombinedParams(scope, params);
      break;
    }
    case 2: {
      /* load scope from param.fbs with meta_version=2 */
      fbs::ParamDeserializer deserializer(reader);
      deserializer.ForwardRead(scope);
      break;
    }
    default:
      LOG(FATAL) << "Unspported model meta_version " << meta_version;
      break;
  }
}

void LoadModelNaiveFromMemory(const std::string &model_buffer,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  // (1)get meta version
  uint16_t meta_version;
  model_parser::StringBufferReader reader(model_buffer);
  reader.Read(&meta_version, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;

  switch (meta_version) {
    case 0:
      LOG(FATAL) << "Paddle-Lite v2.7 has upgraded the naive-buffer model "
                    "format. Please use the OPT to generate a new model. "
                    "Thanks!";
      break;
    case 1:
      LoadModelFbsFromMemory(&reader, scope, cpp_prog, 1);
      break;
    case 2:
      LoadModelFbsFromMemory(&reader, scope, cpp_prog, 2);
      break;
    default:
      LOG(FATAL) << "The model format cannot be recognized. Please make sure "
                    "you use the correct interface and model file.";
      break;
  }
}
///////////////////////////////////////////////////////////////////
// Meta_version=1,2
///////////////////////////////////////////////////////////////////
void LoadModelFbsFromMemory(model_parser::StringBufferReader *reader,
                            Scope *scope,
                            cpp::ProgramDesc *cpp_prog,
                            uint16_t meta_version) {
  // (1)get opt version
  char opt_version[16];
  const uint64_t paddle_version_length = 16 * sizeof(char);
  reader->Read(opt_version, paddle_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);

  // (2)get prog_size and prog_data
  uint64_t prog_size;
  reader->Read(&prog_size, sizeof(uint64_t));
  VLOG(4) << "prog_size:" << prog_size;

  model_parser::Buffer prog_data(prog_size);
  reader->Read(prog_data.data(), prog_size);
#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
  cpp_prog->Init(std::move(prog_data));
#elif LITE_ON_TINY_PUBLISH
  LOG(FATAL) << "Since no data structure of Flatbuffers has been constructed, "
                "the model cannot be loaded.";
#else
  fbs::ProgramDesc program(prog_data);
  TransformProgramDescAnyToCpp(program, cpp_prog);
#endif
  switch (meta_version) {
    case 1: {
      size_t params_size = reader->length() - sizeof(uint16_t) -
                           paddle_version_length - sizeof(uint64_t) - prog_size;
      model_parser::Buffer params_data(params_size);
      reader->Read(params_data.data(), params_size);
      fbs::CombinedParamsDescView params(std::move(params_data));
      fbs::deprecated::SetScopeWithCombinedParams(scope, params);
      break;
    }
    case 2: {
      fbs::ParamDeserializer deserializer(reader);
      deserializer.ForwardRead(scope);
      break;
    }
    default:
      LOG(FATAL) << "Unspported model meta_version " << meta_version;
      break;
  }
  VLOG(4) << "Load model from naive buffer memory successfully";
}

}  // namespace lite
}  // namespace paddle
