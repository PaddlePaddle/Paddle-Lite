# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# See the License for the specific language gov

if(APPLE)
  add_custom_target(publish_inference_cxx_lib ${TARGET}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/*.dylib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    )
  add_custom_target(publish_inference_third_party ${TARGET}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
    COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
  add_dependencies(publish_inference_cxx_lib bundle_full_api)
  add_dependencies(publish_inference_cxx_lib bundle_light_api)
  add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
  add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
  add_dependencies(publish_inference publish_inference_cxx_lib)
  add_dependencies(publish_inference publish_inference_third_party)
elseif(NOT WIN32)
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
  if (LITE_WITH_CUDA)
    add_custom_target(publish_inference_third_party ${TARGET}
      COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
      COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
    add_dependencies(publish_inference publish_inference_third_party)
  endif()
  add_dependencies(publish_inference_cxx_lib bundle_full_api)
  add_dependencies(publish_inference_cxx_lib bundle_light_api)
  add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
  add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
  add_dependencies(publish_inference publish_inference_cxx_lib)
endif()

if(WIN32)
  if(${CMAKE_GENERATOR}  MATCHES "Ninja")
    add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/bin"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/test_model_bin.exe" "${INFER_LITE_PUBLISH_ROOT}/bin"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_use_passes.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/libpaddle_api_full_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/libpaddle_api_light_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    )
  else()
    add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/bin"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api//${CMAKE_BUILD_TYPE}/test_model_bin.exe" "${INFER_LITE_PUBLISH_ROOT}/bin"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_use_passes.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_full_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_light_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    )
  endif()

  add_dependencies(publish_inference_x86_cxx_lib test_model_bin)
  add_dependencies(publish_inference_x86_cxx_lib bundle_full_api)
  add_dependencies(publish_inference_x86_cxx_lib bundle_light_api)
  add_dependencies(publish_inference publish_inference_x86_cxx_lib)

  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)
  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat @ONLY)
  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat @ONLY)

  add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
    COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/CMakeLists.txt.in"
    COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/CMakeLists.txt.in"
    COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/build.bat.in"
    COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/build.bat.in"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
  )
  add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
  add_dependencies(publish_inference_x86_cxx_demos paddle_api_full_bundled eigen3)

else()
  add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
    )
  add_dependencies(publish_inference_x86_cxx_lib test_model_bin)

  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
  configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)   
  add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
    COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
    COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
    COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/*.in"
    COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/*.in"
    COMMAND rm -rf "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
    COMMAND rm -rf "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
  )
  if(WITH_MKL)
  add_custom_command(TARGET publish_inference_x86_cxx_demos POST_BUILD
    COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/")
  endif()
  add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
  add_dependencies(publish_inference_x86_cxx_demos paddle_full_api_shared eigen3)
  add_dependencies(publish_inference publish_inference_x86_cxx_lib)
  add_dependencies(publish_inference publish_inference_x86_cxx_demos)
endif()
