message(STATUS "compile with lite host kernels")

add_kernel(feed_compute_host Host basic SRCS feed_compute.cc DEPS ${lite_kernel_deps})
add_kernel(fetch_compute_host Host basic SRCS fetch_compute.cc DEPS ${lite_kernel_deps})
add_kernel(reshape_compute_host Host basic SRCS reshape_compute.cc DEPS ${lite_kernel_deps} reshape_op)
add_kernel(multiclass_nms_compute_host Host basic SRCS multiclass_nms_compute.cc DEPS ${lite_kernel_deps})

lite_cc_test(test_reshape_compute_host SRCS reshape_compute_test.cc DEPS reshape_compute_host any)
#lite_cc_test(test_multiclass_nms_compute_host SRCS multiclass_nms_compute_test.cc DEPS multiclass_nms_compute_host any)
