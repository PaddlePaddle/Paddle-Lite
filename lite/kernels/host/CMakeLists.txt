message(STATUS "compile with lite host kernels")

# basic kernels

add_kernel(host_basic_kernel Host basic
           SRCS
              feed_compute.cc
              fetch_compute.cc
              range_compute.cc
              reshape_compute.cc
              squeeze_compute.cc
              unsqueeze_compute.cc
              multiclass_nms_compute.cc
              expand_compute.cc
              expand_as_compute.cc
              fill_constant_compute.cc
              fill_constant_batch_size_like_compute.cc
              deformable_conv_compute.cc
              stack_compute.cc
              box_coder_compute.cc
           DEPS
              ${lite_kernel_deps})

# extra kernels
add_kernel(host_extra_kernel Host extra
           SRCS
              unstack_compute.cc
              shape_compute.cc
              is_empty_compute.cc
              crf_decoding_compute.cc
              compare_compute.cc
              logical_compute.cc
              ctc_align_compute.cc
              write_to_array_compute.cc
              read_from_array_compute.cc
              assign_compute.cc
              retinanet_detection_output_compute.cc
              where_index_compute.cc
              print_compute.cc
              while_compute.cc
              conditional_block_compute.cc
              activation_grad_compute.cc
              pixel_shuffle_compute.cc
              one_hot_compute.cc
              uniform_random_compute.cc
              matrix_nms_compute.cc
              sin_compute.cc
              cos_compute.cc
              crop_compute.cc
              crop_tensor_compute.cc
              sequence_unpad_compute.cc
              flatten_compute.cc
              shuffle_channel_compute.cc
              activation_compute.cc
              gather_compute.cc
              pad3d_compute.cc
              select_input_compute.cc
              tensor_array_to_tensor_compute.cc
              expand_v2_compute.cc
           DEPS
              ${lite_kernel_deps})

if(LITE_BUILD_EXTRA AND LITE_WITH_x86)
  lite_cc_test(test_where_index_compute_host SRCS where_index_compute.cc DEPS host_extra_kernel)
  lite_cc_test(test_pixel_shuffle_compute_host SRCS pixel_shuffle_compute.cc DEPS host_extra_kernel)
  lite_cc_test(test_one_hot_compute_host SRCS one_hot_compute_test.cc DEPS host_extra_kernel)
endif()
