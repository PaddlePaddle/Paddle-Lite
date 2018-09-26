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

#if USE_ACL == 1
#include "acl_operator.h"
unsigned int bypass_acl_class_layer =
    (0 | FLAGS_ENABLE_ACL_CONCAT |
     /*0xffffffff |*/ /*FLAGS_ENABLE_ACL_FC |*/ /*FLAGS_ENABLE_ACL_LRN
                                                   |*/
     0);

int enable_schedule = 0;

#ifdef USE_PROFILING

#include "arm_neon.h"

unsigned int acl_log_flags =
    (0 | MASK_LOG_APP_TIME | /*MASK_LOG_ALLOCATE | */  /*MASK_LOG_ALLOCATE | */
     /*MASK_LOG_RUN      | */ /*MASK_LOG_CONFIG   | */ /*MASK_LOG_COPY     | */
     MASK_LOG_ABSVAL | MASK_LOG_BNLL | MASK_LOG_CONV | MASK_LOG_FC |
     MASK_LOG_LRN | MASK_LOG_POOLING | MASK_LOG_RELU | MASK_LOG_SIGMOID |
     MASK_LOG_SOFTMAX | MASK_LOG_TANH | MASK_LOG_LC | MASK_LOG_BN |
     MASK_LOG_CONCAT | 0);
#include <stdio.h>  /* printf */
#include <stdlib.h> /* getenv */
#endif              // USE_PROFILING

static bool force_enable_gpu = false;
bool AclEnableSchedule(int enable) {
  enable_schedule = enable;
  if (enable) {
    force_enable_gpu = true;
  }
  return true;
}
int isScheduleEnable() { return enable_schedule; }

namespace paddle_mobile {
namespace operators {
namespace acl {

bool ACLOperator::init_gpu_env = true;
#ifdef USE_OPENCL
bool ACLOperator::support_opencl_ = false;
bool opencl_is_available() { return arm_compute::opencl_is_available(); }
#elif defined(USE_OPENGLES)
bool ACLOperator::support_opengles_ = false;
#endif
ACLOperator::ACLOperator(bool is_gpu)
    : operator_state_(operator_not_init),
      force_bypass_acl_path_(false),
      target_hint_(TargetHint::DONT_CARE),
      convolution_method_hint_(ConvolutionMethodHint::GEMM),
      _group(1),
      name_(""),
      input_idx_(0),
      output_idx_(0),
      is_gpu_(is_gpu) {
  const char* pBypassACL;
  if (init_gpu_env) {
#ifdef USE_OPENCL
    try {
      if (opencl_is_available()) {
        arm_compute::CLScheduler::get().default_init();
        support_opencl_ = true;
      }
    } catch (std::exception& e) {
      support_opencl_ = false;
    }
#elif defined(USE_OPENGLES)
    try {
      arm_compute::GCScheduler::get().default_init();
      support_opengles_ = true;
    } catch (std::exception& e) {
      support_opengles_ = false;
    }
#endif
    init_gpu_env = false;
  }
  if (force_enable_gpu) is_gpu_ = true;
  pBypassACL = getenv("BYPASSACL");
  if (pBypassACL) {
    unsigned int bacl;
    sscanf(pBypassACL, "%i", &bacl);
    if (bacl != bypass_acl_class_layer) {
      bypass_acl_class_layer = bacl;
      printf("BYPASSACL<%s>\n", pBypassACL);
      printf("BYPASSACL: %x\n", bypass_acl_class_layer);
    }
  }

#ifdef USE_PROFILING
  const char* pLogACL;
  pLogACL = getenv("LOGACL");
  if (pLogACL) {
    unsigned int alf;
    sscanf(pLogACL, "%i", &alf);
    if (alf != acl_log_flags) {
      acl_log_flags = alf;
      printf("LOGACL<%s>\n", pLogACL);
      printf("LOGACL: %x\n", acl_log_flags);
    }
  }
#endif  // USE_PROFILING
  const char* pEnableSchedule;
  pEnableSchedule = getenv("ENABLESCHEDULE");
  if (pEnableSchedule) {
    int bshedule;
    sscanf(pEnableSchedule, "%i", &bshedule);
    if (bshedule != enable_schedule) {
      enable_schedule = bshedule;
      printf("ENABLESCHEDULE<%s>\n", pEnableSchedule);
      printf("ENABLESCHEDULE: %x\n", enable_schedule);
    }
    if (enable_schedule) {
      AclEnableSchedule(1);
    }
  }
}
ACLOperator::~ACLOperator() {}

bool ACLOperator::new_tensor(std::unique_ptr<ACLTensor>& tensor,
                             arm_compute::TensorShape& shape, void* mem,
                             bool commit) {
  auto acl_tensor =
      new ACLTensor(arm_compute::TensorInfo(shape, arm_compute::Format::F32));
  acl_tensor->set_target(getTargetHint());
  acl_tensor->bindmem(mem);
  if (commit) acl_tensor->commit();
  tensor = (std::unique_ptr<ACLTensor>)std::move(acl_tensor);
  return true;
}
bool ACLOperator::new_tensor(std::unique_ptr<ACLSubTensor>& tensor,
                             std::unique_ptr<ACLTensor>& parent,
                             arm_compute::TensorShape& shape,
                             arm_compute::Coordinates& coord) {
  auto acl_tensor = new ACLSubTensor(parent, shape, coord);
  acl_tensor->set_target(getTargetHint());
  tensor = (std::unique_ptr<ACLSubTensor>)std::move(acl_tensor);
  return true;
}

void ACLTensor::commit(TensorType type) {
  settensortype(type);
  if (mem_) {
    if (!allocate_) {
#ifdef USE_PROFILING
      logtime_util log_time(ACL_ALLOCATE_INFO);
#endif  // USE_PROFILING
      allocate();
      allocate_ = true;
    }
    if (type_ != tensor_output) {
      tensor_copy(mem_);
    }
    mem_ = nullptr;
  }
}

int BaseACLTensor::tensor_copy(arm_compute::ITensor* tensor, void* mem,
                               bool toTensor) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_COPY_INFO);
#endif  // USE_PROFILING
  arm_compute::Window window;
  // Iterate through the rows (not each element)
  window.use_tensor_dimensions(tensor->info()->tensor_shape(),
                               /* first_dimension =*/arm_compute::Window::DimY);

  int width = tensor->info()->tensor_shape()[0];
  int height = tensor->info()->tensor_shape()[1];
  int deepth = tensor->info()->tensor_shape()[2];
  map();
  // Create an iterator:
  arm_compute::Iterator it(tensor, window);
  // Except it works for an arbitrary number of dimensions
  if (toTensor) {  // mem->tensor
    arm_compute::execute_window_loop(
        window,
        [&](const arm_compute::Coordinates& id) {
          memcpy(it.ptr(),
                 ((char*)mem) +
                     ((id[3] * (width * height * deepth) +
                       id.z() * (width * height) + id.y() * width + id.x()) *
                      tensor->info()->element_size()),
                 width * tensor->info()->element_size());
        },
        it);
  } else {  // tensor-->mem
    arm_compute::execute_window_loop(
        window,
        [&](const arm_compute::Coordinates& id) {
          memcpy(((char*)mem) + ((id[3] * (width * height * deepth) +
                                  id.z() * (width * height) + id.y() * width) *
                                 tensor->info()->element_size()),
                 it.ptr(), width * tensor->info()->element_size());
        },
        it);
  }
  unmap();

  return 0;
}

}  // namespace acl
}  // namespace operators
}  // namespace paddle_mobile
#endif
