if (NOT LITE_WITH_X86)
  return()
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL)
  return()
endif ()

configure_file (cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
configure_file (warpctc_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/warpctc_lib_path.h)

# source code and dependencies of x86_math static lib
set(X86_MATH_SRC "" CACHE INTERNAL "")
set(X86_MATH_DEPS framework_proto eigen3 CACHE INTERNAL "")

# source code in current directory
FILE(GLOB X86_BASE_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
# source code about jit
FILE(GLOB X86_JIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cc)
FILE(GLOB X86_JIT_REFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jit/refer/*.cc)
if (NOT WIN32 AND NOT APPLE)
FILE(GLOB X86_JIT_GEN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jit/gen/*.cc)
endif()
set(X86_JIT_MORE_SRC "" CACHE INTERNAL "")
add_subdirectory(jit)
# Fluid source file
FILE(GLOB X86_FLUID_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fluid/*.cc)
# detail implementation of x86 math
FILE(GLOB X86_DETAIL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/math/*.cc)
FILE(GLOB X86_DETAIL_AVX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/math/avx/*.cc)
FILE(GLOB X86_DETAIL_SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/math/sse/*.cc)

# Step 1. collect source files
set(X86_MATH_SRC ${X86_MATH_SRC} ${X86_DETAIL_SSE_SRC} ${X86_BASE_SRC} ${X86_FLUID_SRC} ${X86_JIT_SRC} ${X86_JIT_REFER_SRC} ${X86_JIT_MORE_SRC} ${X86_DETAIL_SRC} CACHE INTERNAL "")

# Step2. third party lib
#  2.1 avx
if (WITH_AVX AND AVX_FOUND)
  set(X86_MATH_SRC ${X86_MATH_SRC} ${X86_DETAIL_AVX_SRC})
  if (WIN32)
    set_source_files_properties (${X86_MATH_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX2 /DAVX2 /fp:strict")
  else ()
    set_source_files_properties (${X86_MATH_SRC} PROPERTIES COMPILE_FLAGS "-mfma -mf16c -mavx2")
  endif ()
endif()
#  2.2 xbyak
if(WITH_XBYAK)
  set(X86_MATH_DEPS ${X86_MATH_DEPS} xbyak CACHE INTERNAL "")
  set(X86_MATH_SRC ${X86_MATH_SRC} ${X86_JIT_GEN_SRC} CACHE INTERNAL "")
endif()
#  2.3 mklml
if(WITH_MKL)
  if(WITH_STATIC_MKL)
    set(X86_MATH_DEPS ${X86_MATH_DEPS} ${MKLML_LIBRARIES} CACHE INTERNAL "")
  endif()
endif()

# Step3. Compile into a static lib libx86_math.a
lite_cc_library(x86_math SRCS ${X86_MATH_SRC} DEPS ${X86_MATH_DEPS})
add_dependencies(x86_math eigen3)
