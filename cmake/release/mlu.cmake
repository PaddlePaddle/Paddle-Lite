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

add_custom_target(publish_inference_mlu_cxx_demos ${TARGET}
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mlu_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
  COMMAND cp -r "${CMAKE_BINARY_DIR}/lite/api/libpaddle_full_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
)
add_dependencies(publish_inference_mlu_cxx_demos paddle_full_api_shared)
add_dependencies(publish_inference publish_inference_mlu_cxx_demos)
