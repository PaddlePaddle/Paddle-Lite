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
  
# Build the NNAdapter runtime library and copy it to the publish directory
add_custom_target(publish_inference_nnadapter_runtime
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
  COMMAND cp -r "${CMAKE_BINARY_DIR}/lite/backends/nnadapter/nnadapter/libnnadapter.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
  DEPENDS nnadapter
)
add_dependencies(publish_inference publish_inference_nnadapter_runtime)
# Build the NNAdapter device HAL libraries for all of the specified devices and copy it to the publish directory
foreach(device_name ${NNADAPTER_DEVICES})
  add_custom_target(publish_inference_nnadapter_${device_name}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND cp -r "${CMAKE_BINARY_DIR}/lite/backends/nnadapter/nnadapter/driver/*/lib${device_name}.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    DEPENDS ${device_name}
    )
  add_dependencies(publish_inference publish_inference_nnadapter_${device_name})
endforeach()
