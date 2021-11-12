# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

if(LITE_WITH_SW)
  add_custom_target(publish_inference_cxx_lib ${TARGET}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/*.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
  )
  add_dependencies(publish_inference_cxx_lib bundle_full_api)
  add_dependencies(publish_inference_cxx_lib bundle_light_api)
  add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
  add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
  add_dependencies(publish_inference publish_inference_cxx_lib)
  add_dependencies(publish_inference test_model_bin)
endif()

if (NOT XPU_SDK_ROOT)
add_custom_command(TARGET publish_inference POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${THIRD_PARTY_PATH}/install/xpu" "${INFER_LITE_PUBLISH_ROOT}/third_party/xpu"
)
endif()
