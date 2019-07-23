if (NOT LITE_WITH_FPGA)
    return()
endif()

set(LITE_FPGA_KD_PATH "${PADDLE_SOURCE_DIR}/lite/fpga/KD")
set(LITE_FPGA_PATH "${PADDLE_SOURCE_DIR}/lite/fpga")

message("fpga_kd_path ${LITE_FPGA_KD_PATH}")
message("fpga_path ${LITE_FPGA_PATH}")
file(GLOB_RECURSE KD_CPP *.cpp *.cc)
file(GLOB FPGA_CPP "${LITE_FPGA_PATH}/*.cc")

cc_library(kernel_fpga SRCS ${KD_CPP} ${FPGA_CPP})
cc_library(fpga_target_wrapper SRCS ${LITE_FPGA_PATH}/target_wrapper.cc DEPS kernel_fpga)
