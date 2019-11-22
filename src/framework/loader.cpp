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

#include "framework/loader.h"

#include "framework/lod_tensor.h"
#include "framework/program/program-optimize/program_optimize.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_image.h"
#endif
#ifdef MODEL_SECU
#include <stdlib.h>
#include <stdio.h>
#include "seco/seco.h"  
#include <iostream> 
#endif

namespace paddle_mobile {
namespace framework {

template <typename Device, typename T>
void Loader<Device, T>::InitMemoryFromProgram(
    const std::shared_ptr<ProgramDesc> &originProgramDesc,
    const std::shared_ptr<Scope> &scope) {
  for (const auto &block : originProgramDesc.get()->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = scope.get()->Var(var_desc->Name());
      if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable()) {
          auto dim = var_desc->Tensor_desc().Dims();
          auto tensor = var->GetMutable<LoDTensor>();
          tensor->Resize(make_ddim(dim));
        } else {
          auto dim = var_desc->Tensor_desc().Dims();
          if (dim.size() == 0) {
            auto tensor = var->GetMutable<LoDTensor>();
            framework::DDim dDim = {0};
            tensor->Resize(dDim);
          } else {
            for (auto &d : dim) {
              if (d < 0) {
                d *= -1;
              }
            }
            auto tensor = var->GetMutable<LoDTensor>();
            tensor->Resize(make_ddim(dim));
          }
        }
      } else {
        // TODO(codeWorm)
      }
    }
  }
}

#ifdef PADDLE_MOBILE_CL
template <>
void Loader<GPU_CL, float>::InitMemoryFromProgram(
    const std::shared_ptr<ProgramDesc> &originProgramDesc,
    const std::shared_ptr<Scope> &scope) {
  for (const auto &block : originProgramDesc.get()->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = scope.get()->Var(var_desc->Name());
      if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable()) {
          auto dim = var_desc->Tensor_desc().Dims();
          auto cl_image = var->GetMutable<framework::CLImage>();
          cl_image->Resize(make_ddim(dim));
        } else {
          auto dim = var_desc->Tensor_desc().Dims();
          PADDLE_MOBILE_ENFORCE(dim.size() > 0, "dim size is 0");
          dim[0] = 1;
          auto cl_image = var->GetMutable<framework::CLImage>();
          cl_image->Resize(make_ddim(dim));
        }
      } else {
        // TODO(codeWorm)
      }
    }
  }
}
template <>
const Program<GPU_CL, float> Loader<GPU_CL, float>::LoadCombinedMemory(
    size_t read_size, const uint8_t *buf, size_t combined_params_len,
    uint8_t *combined_params_buf, bool optimize, bool quantification) {
  bool can_add_split = false;

  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  PADDLE_MOBILE_ENFORCE(buf != nullptr, "read from __model__ is null");

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      nullptr, read_size, buf);
  //
  PADDLE_MOBILE_ENFORCE(c_program != nullptr, "program is null");
  //
  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
  //

  auto originProgramDesc = std::make_shared<ProgramDesc>(c_program);

  Program<GPU_CL, float> program;
  program.combined = true;
  program.originProgram = originProgramDesc;
  program.quantification = quantification;
  program.combined_params_len = combined_params_len;
  program.combined_params_buf = combined_params_buf;

  auto scope = std::make_shared<Scope>();
  program.scope = scope;
  InitMemoryFromProgram(originProgramDesc, scope);
  if (optimize) {
    ProgramOptimize program_optimize;
    program.optimizeProgram =
        program_optimize.FusionOptimize(originProgramDesc, can_add_split);
    if (!program.optimizeProgram) {
      program.optimizeProgram = originProgramDesc;
    }
  }
  if (optimize) {
    program.optimizeProgram->Description("optimize: ");
  } else {
    originProgramDesc->Description("program: ");
  }
  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program,
                                                               nullptr);
  return program;
}

#endif

/**
 * fusion and print someinfos
 * @tparam Device
 * @tparam P
 * @param optimize
 * @param can_add_split
 * @param program
 * @param originProgramDesc
 */
template <typename Device, typename T>
void FusionAndPrintInfos(
    bool optimize, bool can_add_split, Program<Device, T> *program,
    const std::shared_ptr<ProgramDesc> &originProgramDesc) {
  if (optimize) {
    ProgramOptimize program_optimize;
    program->optimizeProgram =
        program_optimize.FusionOptimize(originProgramDesc, can_add_split);
    if (!program->optimizeProgram) {
      program->optimizeProgram = originProgramDesc;
    }
  }
  if (optimize) {
    program->optimizeProgram->Description("optimize: ");
  } else {
    originProgramDesc->Description("program: ");
  }
}

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
#ifdef MODEL_SECU
   std::cout<<"begin readbuffer"<<std::endl;
   Seco *seco = new Seco();
   unsigned char pub_key[512];
   int value = seco->read_pubkey_from_chip(pub_key);
   if (value!=0){
        delete seco;
        LOG(kLOG_ERROR) << "read chip encrypt key error";
        return 0;
    }
    long out_length = 0;
    std::string model_file(file_name);
    value = seco->parse_model(pub_key, model_file, out, out_length);  
    delete seco;
    if (value!=0){
        LOG(kLOG_ERROR) << "read buffer error code is:" << value;
        return 0;  
    }
    std::cout<<"out_length is:"<<out_length<<std::endl;
    size_t cur_len = out_length;
    return cur_len;
#else
  FILE *fp;
  fp = fopen(file_name, "rb");
  PADDLE_MOBILE_ENFORCE(fp != NULL, " %s open failed !", file_name);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);

  DLOG << "model size: " << size;

  *out = reinterpret_cast<uint8_t *>(malloc(size));

  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
#endif
}

template <typename Device, typename T>
const Program<Device, T> Loader<Device, T>::Load(const std::string &dirname,
                                                 bool optimize,
                                                 bool quantification,
                                                 bool can_add_split) {
  auto program = this->LoadProgram(dirname + "/__model__", optimize,
                                   quantification, can_add_split);
  program.model_path = dirname;
  return program;
}

template <typename Device, typename T>
const Program<Device, T> Loader<Device, T>::Load(const std::string &model_path,
                                                 const std::string &para_path,
                                                 bool optimize,
                                                 bool quantification) {
  auto program = this->LoadProgram(model_path, optimize, quantification);

  program.para_path = para_path;
  program.combined = true;
  program.quantification = quantification;
  return program;
}

template <typename Device, typename T>
const Program<Device, T> Loader<Device, T>::LoadProgram(
    const std::string &model_path, bool optimize, bool quantification,
    bool can_add_split) {
  std::string model_filename = model_path;
  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  uint8_t *buf = NULL;
  size_t read_size = ReadBuffer(model_filename.c_str(), &buf);

  PADDLE_MOBILE_ENFORCE(buf != NULL, "read from __model__ is null");

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      NULL, read_size, buf);
  //
  PADDLE_MOBILE_ENFORCE(c_program != NULL, "program is null");
  //
  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
  //
  auto originProgramDesc = std::make_shared<ProgramDesc>(c_program);

  Program<Device, T> program;
  program.originProgram = originProgramDesc;
  program.quantification = quantification;
  program.combined_params_len = 0;
  program.combined_params_buf = nullptr;
  auto scope = std::make_shared<Scope>();
  program.scope = scope;

  // use  originProgramDesc and scope to init tensors
  InitMemoryFromProgram(originProgramDesc, scope);
  // perform fusion and print infos
  FusionAndPrintInfos(optimize, can_add_split, &program, originProgramDesc);

  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program, NULL);
  free(buf);
  return program;
}

template <typename Device, typename T>
const Program<Device, T> Loader<Device, T>::LoadCombinedMemory(
    size_t read_size, const uint8_t *buf, size_t combined_params_len,
    uint8_t *combined_params_buf, bool optimize, bool quantification) {
  bool can_add_split = false;

  PaddleMobile__Framework__Proto__ProgramDesc *c_program;
  PADDLE_MOBILE_ENFORCE(buf != nullptr, "read from __model__ is null");

  c_program = paddle_mobile__framework__proto__program_desc__unpack(
      nullptr, read_size, buf);
  //
  PADDLE_MOBILE_ENFORCE(c_program != nullptr, "program is null");
  //
  DLOG << "n_ops: " << (*c_program->blocks)->n_ops;
  //

  auto originProgramDesc = std::make_shared<ProgramDesc>(c_program);

  Program<Device, T> program;
  program.combined = true;
  program.originProgram = originProgramDesc;
  program.quantification = quantification;
  program.combined_params_len = combined_params_len;
  program.combined_params_buf = combined_params_buf;

  auto scope = std::make_shared<Scope>();
  program.scope = scope;
  InitMemoryFromProgram(originProgramDesc, scope);
  FusionAndPrintInfos(optimize, can_add_split, &program, originProgramDesc);
  paddle_mobile__framework__proto__program_desc__free_unpacked(c_program,
                                                               nullptr);
  return program;
}

template class Loader<CPU, float>;

template class Loader<FPGA, float>;

template class Loader<GPU_MALI, float>;

template class Loader<GPU_CL, float>;

}  // namespace framework
}  // namespace paddle_mobile
