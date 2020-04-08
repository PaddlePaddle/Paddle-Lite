message(STATUS "compile with lite host kernels")

add_kernel(feed_compute_host Host basic SRCS feed_compute.cc DEPS ${lite_kernel_deps})
add_kernel(fetch_compute_host Host basic SRCS fetch_compute.cc DEPS ${lite_kernel_deps})
add_kernel(reshape_compute_host Host basic SRCS reshape_compute.cc DEPS ${lite_kernel_deps} reshape_op)
add_kernel(multiclass_nms_compute_host Host basic SRCS multiclass_nms_compute.cc DEPS ${lite_kernel_deps})
add_kernel(crf_decoding_compute_host Host extra SRCS crf_decoding_compute.cc DEPS ${lite_kernel_deps})
add_kernel(ctc_align_compute_host Host extra SRCS ctc_align_compute.cc DEPS ${lite_kernel_deps})
