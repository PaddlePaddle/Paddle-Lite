function(USE_JITKERNEL_MORE_LITE TARGET TYPE)
    file(APPEND ${jit_file} "USE_JITKERNEL_MORE_LITE(${TARGET} ${TYPE});\n")
endfunction()

IF(WITH_LASX)
  ADD_SUBDIRECTORY(intrinsic)
ENDIF()

# mix should be last
add_subdirectory(mix)
