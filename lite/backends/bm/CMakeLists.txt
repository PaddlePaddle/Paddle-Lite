if (NOT LITE_WITH_BM)
    return()
endif()

#lite_cc_library(target_wrapper_bm SRCS target_wrapper.cc DEPS ${bm_runtime_libs})

add_library(target_wrapper_bm STATIC target_wrapper.cc)
target_link_libraries(target_wrapper_bm ${BM_SDK_ROOT}/lib/bmcompiler/libbmcompiler.so ${BM_SDK_ROOT}/lib/bmnn/pcie/libbmcpu.so ${BM_SDK_ROOT}/lib/bmnn/pcie/libbmlib.so ${BM_SDK_ROOT}/lib/bmnn/pcie/libbmrt.so)
