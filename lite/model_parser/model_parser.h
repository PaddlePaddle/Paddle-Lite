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

// This file contains model format related operations, such as load a model,
// parse an operator definitions and so on.

#pragma once
#include <memory>
#include <string>
#include <vector>
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/core/framework.pb.h"
#endif
#include "lite/core/scope.h"
#include "lite/core/variable.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {

#ifndef LITE_ON_TINY_PUBLISH
// Read a __model__ file.
std::unique_ptr<framework::proto::ProgramDesc> LoadProgram(
    const std::string& path, bool program_from_memory = false);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path, Variable* out);

void LoadCombinedParamsPb(const std::string& path,
                          lite::Scope* scope,
                          const cpp::ProgramDesc& prog,
                          bool params_from_memory = false);

// Read a model and files of parameters in pb format.
void LoadModelPb(const std::string& model_dir,
                 const std::string& model_file,
                 const std::string& param_file,
                 Scope* scope,
                 cpp::ProgramDesc* prog,
                 bool combined = false,
                 bool model_from_memory = false);

// Save a model and files of parameters in pb format.
void SaveModelPb(const std::string& model_dir,
                 const Scope& scope,
                 const cpp::ProgramDesc& prog,
                 bool combined = false);

void SaveCombinedParamsPb(const std::string& path,
                          const lite::Scope& exec_scope,
                          const cpp::ProgramDesc& prog);

// Serialize tensors to ostream.
void SerializeTensor(std::ostream& os,
                     const lite::Scope& scope,
                     const std::string& var);

// LoDTensor to ostream
void TensorToStream(std::ostream& os, const lite::Tensor& tensor);
void TensorFromStream(std::istream& is, lite::Tensor* tensor);
void ReadBinaryFile(const std::string& filename, std::string* contents);

// For naive buffer
void SaveParamNaive(const std::string& path,
                    const lite::Scope& exec_scope,
                    const std::string& var_name);

void SaveCombinedParamsNaive(const std::string& path,
                             const lite::Scope& exec_scope,
                             const cpp::ProgramDesc& cpp_prog);

void SaveModelNaive(const std::string& model_dir,
                    const Scope& exec_scope,
                    const cpp::ProgramDesc& cpp_prog,
                    bool combined = true);

void SaveModelFbs(const std::string& model_dir,
                  const Scope& exec_scope,
                  const cpp::ProgramDesc& cpp_prog);

void LoadModelFbsFromFile(const std::string& filename,
                          Scope* scope,
                          cpp::ProgramDesc* cpp_prog);
#endif  // LITE_ON_TINY_PUBLISH

void LoadParamNaive(const std::string& path,
                    lite::Scope* scope,
                    const std::string& name);

// warning:this old inference will be abandened in release/v3.0.0
// and LoadModelNaiveFromFile is suggested.
void LoadModelNaive(const std::string& model_dir,
                    lite::Scope* scope,
                    cpp::ProgramDesc* prog,
                    bool combined = true);
void LoadModelNaiveFromFile(const std::string& filename,
                            lite::Scope* scope,
                            cpp::ProgramDesc* prog);
void LoadModelNaiveFromMemory(const std::string& model_buffer,
                              const std::string& param_buffer,
                              lite::Scope* scope,
                              cpp::ProgramDesc* cpp_prog);
void LoadModelNaiveFromMemory(const std::string& model_buffer,
                              lite::Scope* scope,
                              cpp::ProgramDesc* cpp_prog);

}  // namespace lite
}  // namespace paddle
