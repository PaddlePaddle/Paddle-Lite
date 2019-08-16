
set(ARCH "armv7-a")

set(FLOAT_ABI "softfp" CACHE STRING "-mfloat-api chosen")
set_property(CACHE FLOAT_ABI PROPERTY STRINGS "softfp" "soft" "hard")

set(FPU "neon")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${ARCH} -mfloat-abi=${FLOAT_ABI} -mfpu=${FPU}")
