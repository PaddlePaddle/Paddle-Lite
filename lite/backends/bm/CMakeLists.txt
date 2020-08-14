if (NOT LITE_WITH_BM)
    return()
endif()

add_library(target_wrapper_bm STATIC target_wrapper.cc)
target_link_libraries(target_wrapper_bm -Wl,-rpath,${BM_SDK_ROOT}/lib/bmcompiler:${BM_SDK_ROOT}/lib/bmnn/pcie -L${BM_SDK_ROOT}/lib/bmcompiler -L${BM_SDK_ROOT}/lib/bmnn/pcie -lbmcompiler -lbmcpu -lbmlib -lbmrt)
