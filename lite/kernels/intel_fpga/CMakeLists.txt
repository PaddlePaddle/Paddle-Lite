if(LITE_WITH_INTEL_FPGA)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

set(intel_fpga_deps ${lite_kernel_deps} ${intel_fpga_runtime_libs})

add_kernel(conv_depthwise_intel_fpga INTEL_FPGA basic SRCS conv_depthwise.cc)
add_kernel(conv_gemmlike_intel_fpga INTEL_FPGA basic SRCS conv_gemmlike.cc)
add_kernel(conv_compute_intel_fpga INTEL_FPGA basic SRCS conv_compute.cc)

add_dependencies(kenrels ${intel_fpga_runtime_libs})
target_link_libraries(kernels ${intel_fpga_runtime_libs})
