
function(USE_JITKERNEL_MORE_LITE TARGET TYPE)
    file(APPEND ${jit_file} "USE_JITKERNEL_MORE_LITE(${TARGET} ${TYPE});\n")
endfunction()

# enable it latter
 if(WITH_MKLML)
     add_subdirectory(mkl)
 endif()

if(WITH_AVX)
    add_subdirectory(intrinsic)
endif()

# mix should be last
add_subdirectory(mix)

set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} PARENT_SCOPE)
