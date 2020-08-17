if (NOT LITE_WITH_BM)
    return()
endif()

add_library(target_wrapper_bm STATIC target_wrapper.cc)
target_link_libraries(target_wrapper_bm -Wl,-rpath,${BM_SDK_CPLIB_RPATH}:${BM_SDK_LIB_RPATH} -L${BM_SDK_CPLIB_RPATH} -L${BM_SDK_LIB_RPATH} -lbmcompiler -lbmcpu -lbmlib -lbmrt)
