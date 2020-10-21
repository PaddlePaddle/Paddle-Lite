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

#include "NeuronAdapter.h"  // NOLINT
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class NeuronAdapter final {
 public:
  static NeuronAdapter *Global();
  // Platform APIs
  using Neuron_getVersion_Type = int (*)(uint32_t *);
  using NeuronModel_create_Type = int (*)(NeuronModel **);
  using NeuronModel_free_Type = void (*)(NeuronModel *);
  using NeuronModel_finish_Type = int (*)(NeuronModel *);
  using NeuronModel_addOperand_Type = int (*)(NeuronModel *,
                                              const NeuronOperandType *);
  using NeuronModel_setOperandValue_Type = int (*)(NeuronModel *,
                                                   int32_t,
                                                   const void *,
                                                   size_t);
  using NeuronModel_setOperandSymmPerChannelQuantParams_Type =
      int (*)(NeuronModel *, int32_t, const NeuronSymmPerChannelQuantParams *);
  using NeuronModel_addOperation_Type = int (*)(NeuronModel *,
                                                NeuronOperationType,
                                                uint32_t,
                                                const uint32_t *,
                                                uint32_t,
                                                const uint32_t *);
  using NeuronModel_addOperationExtension_Type = int (*)(NeuronModel *,
                                                         const char *,
                                                         const char *,
                                                         const NeuronDevice *,
                                                         uint32_t,
                                                         const uint32_t *,
                                                         uint32_t,
                                                         const uint32_t *);
  using NeuronModel_identifyInputsAndOutputs_Type = int (*)(
      NeuronModel *, uint32_t, const uint32_t *, uint32_t, const uint32_t *);
  using NeuronModel_restoreFromCompiledNetwork_Type =
      int (*)(NeuronModel **, NeuronCompilation **, const void *, const size_t);
  using NeuronCompilation_create_Type = int (*)(NeuronModel *,
                                                NeuronCompilation **);
  using NeuronCompilation_free_Type = void (*)(NeuronCompilation *);
  using NeuronCompilation_finish_Type = int (*)(NeuronCompilation *);
  using NeuronCompilation_setCaching_Type = int (*)(NeuronCompilation *,
                                                    const char *,
                                                    const uint8_t *);
  using NeuronCompilation_createForDevices_Type =
      int (*)(NeuronModel *,
              const NeuronDevice *const *,
              uint32_t,
              NeuronCompilation **);
  using NeuronCompilation_storeCompiledNetwork_Type =
      int (*)(NeuronCompilation *, void *, const size_t);
  using NeuronCompilation_getCompiledNetworkSize_Type =
      int (*)(NeuronCompilation *, size_t *);
  using NeuronExecution_create_Type = int (*)(NeuronCompilation *,
                                              NeuronExecution **);
  using NeuronExecution_free_Type = void (*)(NeuronExecution *);
  using NeuronExecution_setInput_Type = int (*)(NeuronExecution *,
                                                int32_t,
                                                const NeuronOperandType *,
                                                const void *,
                                                size_t);
  using NeuronExecution_setOutput_Type = int (*)(
      NeuronExecution *, int32_t, const NeuronOperandType *, void *, size_t);
  using NeuronExecution_compute_Type = int (*)(NeuronExecution *);
  using Neuron_getDeviceCount_Type = int (*)(uint32_t *);
  using Neuron_getDevice_Type = int (*)(uint32_t, NeuronDevice **);
  using NeuronDevice_getName_Type = int (*)(const NeuronDevice *,
                                            const char **);

  Neuron_getVersion_Type Neuron_getVersion() {
    CHECK(Neuron_getVersion_ != nullptr) << "Cannot load "
                                            "Neuron_"
                                            "getVersion!";
    return Neuron_getVersion_;
  }

  NeuronModel_restoreFromCompiledNetwork_Type
  NeuronModel_restoreFromCompiledNetwork() {
    CHECK(NeuronModel_restoreFromCompiledNetwork_ != nullptr)
        << "Cannot load "
           "NeuronModel_"
           "restoreFromCompil"
           "edNetwork!";
    return NeuronModel_restoreFromCompiledNetwork_;
  }

  NeuronModel_create_Type NeuronModel_create() {
    CHECK(NeuronModel_create_ != nullptr) << "Cannot load "
                                             "NeuronModel_"
                                             "create!";
    return NeuronModel_create_;
  }

  NeuronModel_free_Type NeuronModel_free() {
    CHECK(NeuronModel_free_ != nullptr) << "Cannot load "
                                           "NeuronModel_"
                                           "free!";
    return NeuronModel_free_;
  }

  NeuronModel_finish_Type NeuronModel_finish() {
    CHECK(NeuronModel_finish_ != nullptr) << "Cannot load "
                                             "NeuronModel_"
                                             "finish!";
    return NeuronModel_finish_;
  }

  NeuronModel_addOperand_Type NeuronModel_addOperand() {
    CHECK(NeuronModel_addOperand_ != nullptr) << "Cannot load "
                                                 "NeuronModel_"
                                                 "addOperand!";
    return NeuronModel_addOperand_;
  }

  NeuronModel_setOperandValue_Type NeuronModel_setOperandValue() {
    CHECK(NeuronModel_setOperandValue_ != nullptr) << "Cannot load "
                                                      "NeuronModel_"
                                                      "setOperandValue!";
    return NeuronModel_setOperandValue_;
  }

  NeuronModel_setOperandSymmPerChannelQuantParams_Type
  NeuronModel_setOperandSymmPerChannelQuantParams() {
    CHECK(NeuronModel_setOperandSymmPerChannelQuantParams_ != nullptr)
        << "Cannot load "
           "NeuronModel_"
           "setOperandSymmPer"
           "ChannelQuantParam"
           "s!";
    return NeuronModel_setOperandSymmPerChannelQuantParams_;
  }

  NeuronModel_addOperation_Type NeuronModel_addOperation() {
    CHECK(NeuronModel_addOperation_ != nullptr) << "Cannot load "
                                                   "NeuronModel_"
                                                   "addOperation!";
    return NeuronModel_addOperation_;
  }

  NeuronModel_addOperationExtension_Type NeuronModel_addOperationExtension() {
    CHECK(NeuronModel_addOperationExtension_ != nullptr) << "Cannot load "
                                                            "NeuronModel_"
                                                            "addOperationExten"
                                                            "sion!";
    return NeuronModel_addOperationExtension_;
  }

  NeuronModel_identifyInputsAndOutputs_Type
  NeuronModel_identifyInputsAndOutputs() {
    CHECK(NeuronModel_identifyInputsAndOutputs_ != nullptr)
        << "Cannot load "
           "NeuronModel_"
           "identifyInputsAnd"
           "Outputs!";
    return NeuronModel_identifyInputsAndOutputs_;
  }

  NeuronCompilation_create_Type NeuronCompilation_create() {
    CHECK(NeuronCompilation_create_ != nullptr) << "Cannot load "
                                                   "NeuronCompilation"
                                                   "_create!";
    return NeuronCompilation_create_;
  }

  NeuronCompilation_free_Type NeuronCompilation_free() {
    CHECK(NeuronCompilation_free_ != nullptr) << "Cannot load "
                                                 "NeuronCompilation"
                                                 "_free!";
    return NeuronCompilation_free_;
  }

  NeuronCompilation_finish_Type NeuronCompilation_finish() {
    CHECK(NeuronCompilation_finish_ != nullptr) << "Cannot load "
                                                   "NeuronCompilation"
                                                   "_finish!";
    return NeuronCompilation_finish_;
  }

  NeuronCompilation_setCaching_Type NeuronCompilation_setCaching() {
    CHECK(NeuronCompilation_setCaching_ != nullptr) << "Cannot load "
                                                       "NeuronCompilation"
                                                       "_setCaching!";
    return NeuronCompilation_setCaching_;
  }

  NeuronCompilation_createForDevices_Type NeuronCompilation_createForDevices() {
    CHECK(NeuronCompilation_createForDevices_ != nullptr) << "Cannot load "
                                                             "NeuronCompilation"
                                                             "_createForDevices"
                                                             "!";
    return NeuronCompilation_createForDevices_;
  }

  NeuronCompilation_storeCompiledNetwork_Type
  NeuronCompilation_storeCompiledNetwork() {
    CHECK(NeuronCompilation_storeCompiledNetwork_ != nullptr)
        << "Cannot load "
           "NeuronCompilation"
           "_storeCompiledNet"
           "work!";
    return NeuronCompilation_storeCompiledNetwork_;
  }

  NeuronCompilation_getCompiledNetworkSize_Type
  NeuronCompilation_getCompiledNetworkSize() {
    CHECK(NeuronCompilation_getCompiledNetworkSize_ != nullptr)
        << "Cannot load "
           "NeuronCompilation"
           "_getCompiledNetwo"
           "rkSize!";
    return NeuronCompilation_getCompiledNetworkSize_;
  }

  NeuronExecution_create_Type NeuronExecution_create() {
    CHECK(NeuronExecution_create_ != nullptr) << "Cannot load "
                                                 "NeuronExecution_"
                                                 "create!";
    return NeuronExecution_create_;
  }

  NeuronExecution_free_Type NeuronExecution_free() {
    CHECK(NeuronExecution_free_ != nullptr) << "Cannot load "
                                               "NeuronExecution_"
                                               "free!";
    return NeuronExecution_free_;
  }

  NeuronExecution_setInput_Type NeuronExecution_setInput() {
    CHECK(NeuronExecution_setInput_ != nullptr) << "Cannot loadcl "
                                                   "NeuronExecution_"
                                                   "setInput!";
    return NeuronExecution_setInput_;
  }

  NeuronExecution_setOutput_Type NeuronExecution_setOutput() {
    CHECK(NeuronExecution_setOutput_ != nullptr) << "Cannot load "
                                                    "NeuronExecution_"
                                                    "setOutput!";
    return NeuronExecution_setOutput_;
  }

  NeuronExecution_compute_Type NeuronExecution_compute() {
    CHECK(NeuronExecution_compute_ != nullptr) << "Cannot load "
                                                  "NeuronExecution_"
                                                  "compute!";
    return NeuronExecution_compute_;
  }

  Neuron_getDeviceCount_Type Neuron_getDeviceCount() {
    CHECK(Neuron_getDeviceCount_ != nullptr) << "Cannot load "
                                                "Neuron_"
                                                "getDeviceCount!";
    return Neuron_getDeviceCount_;
  }

  Neuron_getDevice_Type Neuron_getDevice() {
    CHECK(Neuron_getDevice_ != nullptr) << "Cannot load "
                                           "Neuron_"
                                           "getDevice!";
    return Neuron_getDevice_;
  }

  NeuronDevice_getName_Type NeuronDevice_getName() {
    CHECK(NeuronDevice_getName_ != nullptr) << "Cannot load "
                                               "NeuronDevice_"
                                               "getName!";
    return NeuronDevice_getName_;
  }

 private:
  NeuronAdapter();
  NeuronAdapter(const NeuronAdapter &) = delete;
  NeuronAdapter &operator=(const NeuronAdapter &) = delete;
  bool InitHandle();
  void InitFunctions();
  void *handle_{nullptr};
  Neuron_getVersion_Type Neuron_getVersion_{nullptr};
  NeuronModel_create_Type NeuronModel_create_{nullptr};
  NeuronModel_free_Type NeuronModel_free_{nullptr};
  NeuronModel_finish_Type NeuronModel_finish_{nullptr};
  NeuronModel_addOperand_Type NeuronModel_addOperand_{nullptr};
  NeuronModel_setOperandValue_Type NeuronModel_setOperandValue_{nullptr};
  NeuronModel_setOperandSymmPerChannelQuantParams_Type
      NeuronModel_setOperandSymmPerChannelQuantParams_{nullptr};
  NeuronModel_addOperation_Type NeuronModel_addOperation_{nullptr};
  NeuronModel_addOperationExtension_Type NeuronModel_addOperationExtension_{
      nullptr};
  NeuronModel_identifyInputsAndOutputs_Type
      NeuronModel_identifyInputsAndOutputs_{nullptr};
  NeuronModel_restoreFromCompiledNetwork_Type
      NeuronModel_restoreFromCompiledNetwork_{nullptr};
  NeuronCompilation_create_Type NeuronCompilation_create_{nullptr};
  NeuronCompilation_free_Type NeuronCompilation_free_{nullptr};
  NeuronCompilation_finish_Type NeuronCompilation_finish_{nullptr};
  NeuronCompilation_setCaching_Type NeuronCompilation_setCaching_{nullptr};
  NeuronCompilation_createForDevices_Type NeuronCompilation_createForDevices_{
      nullptr};
  NeuronCompilation_storeCompiledNetwork_Type
      NeuronCompilation_storeCompiledNetwork_{nullptr};
  NeuronCompilation_getCompiledNetworkSize_Type
      NeuronCompilation_getCompiledNetworkSize_{nullptr};
  NeuronExecution_create_Type NeuronExecution_create_{nullptr};
  NeuronExecution_free_Type NeuronExecution_free_{nullptr};
  NeuronExecution_setInput_Type NeuronExecution_setInput_{nullptr};
  NeuronExecution_setOutput_Type NeuronExecution_setOutput_{nullptr};
  NeuronExecution_compute_Type NeuronExecution_compute_{nullptr};
  Neuron_getDeviceCount_Type Neuron_getDeviceCount_{nullptr};
  Neuron_getDevice_Type Neuron_getDevice_{nullptr};
  NeuronDevice_getName_Type NeuronDevice_getName_{nullptr};
};
}  // namespace lite
}  // namespace paddle
