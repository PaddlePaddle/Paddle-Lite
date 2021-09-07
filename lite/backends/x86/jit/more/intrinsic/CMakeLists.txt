file(GLOB jit_kernel_cc_intrinsic "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(X86_JIT_MORE_SRC ${X86_JIT_MORE_SRC} ${jit_kernel_cc_intrinsic} CACHE INTERNAL "")

# use mkl kernels by name and type
USE_JITKERNEL_MORE_LITE(kCRFDecoding, intrinsic)
USE_JITKERNEL_MORE_LITE(kLayerNorm, intrinsic)
