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

add_custom_target(publish_inference_opencl ${TARGET}
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/opencl"
  COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/backends/opencl/cl_kernel" "${INFER_LITE_PUBLISH_ROOT}/opencl"
  )
if(NOT LITE_ON_TINY_PUBLISH)
  add_dependencies(publish_inference_cxx_lib publish_inference_opencl)
else()
  add_dependencies(tiny_publish_cxx_lib publish_inference_opencl)
endif()
