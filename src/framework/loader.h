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

#include <string>

#include "common/types.h"
#include "framework/program/program.h"
#ifdef MODEL_SECU
#include "seco/seco.h" 
#endif

namespace paddle_mobile {
namespace framework {

template <typename Device = CPU, typename T = float>
class Loader {
 public:
  /*
   * @b load separate format fluid model
   * @b 加载分开存储的fluid模型
   * */
  const Program<Device, T> Load(const std::string &dirname,
                                bool optimize = false,
                                bool quantification = false,
                                bool can_add_split = false);

  /*
   * @b load combine format fluid mode
   * @b 加载统一存储的fluid模型
   * */
  const Program<Device, T> Load(const std::string &model_path,
                                const std::string &para_path,
                                bool optimize = false,
                                bool quantification = false);

  const Program<Device, T> LoadCombinedMemory(size_t model_len,
                                              const uint8_t *model_buf,
                                              size_t combined_params_len,
                                              uint8_t *combined_params_buf,
                                              bool optimize = false,
                                              bool quantification = false);

 private:
  const Program<Device, T> LoadProgram(const std::string &model_path,
                                       bool optimize = false,
                                       bool quantification = false,
                                       bool can_add_split = false);

  void InitMemoryFromProgram(
      const std::shared_ptr<ProgramDesc> &originProgramDesc,
      const std::shared_ptr<Scope> &scope);
};

}  // namespace framework
}  // namespace paddle_mobile
