lite_cc_library(subgraph_bridge_scale_op_mlu SRCS scale_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_interp_op_mlu SRCS interpolate_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_concat_op_mlu SRCS concat_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_transpose_op_mlu SRCS transpose_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_dropout_op_mlu SRCS dropout_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_slice_op_mlu SRCS slice_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_split_op_mlu SRCS split_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_cast_op_mlu SRCS cast_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_layout_op_mlu SRCS layout_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_argmax_op_mlu SRCS argmax_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_squeeze_op_mlu SRCS squeeze_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_reshape_op_mlu SRCS reshape_op.cc DEPS ${subgraph_bridge_deps_mlu})
lite_cc_library(subgraph_bridge_flatten_op_mlu SRCS flatten_op.cc DEPS ${subgraph_bridge_deps_mlu})
set(mlu_subgraph_bridges
        core
        subgraph_bridge_transpose_op_mlu
        subgraph_bridge_batch_norm_op_mlu
        subgraph_bridge_scale_op_mlu
        subgraph_bridge_interp_op_mlu
        subgraph_bridge_concat_op_mlu
        subgraph_bridge_dropout_op_mlu
        subgraph_bridge_slice_op_mlu
        subgraph_bridge_split_op_mlu
        subgraph_bridge_cast_op_mlu
        subgraph_bridge_layout_op_mlu
        subgraph_bridge_argmax_op_mlu
        subgraph_bridge_squeeze_op_mlu
        subgraph_bridge_reshape_op_mlu
        subgraph_bridge_flatten_op_mlu
        CACHE INTERNAL "mlu_subgraph_bridges")


if (LITE_BUILD_EXTRA)
  lite_cc_library(subgraph_bridge_lrn_op_mlu SRCS lrn_op.cc DEPS ${subgraph_bridge_deps_mlu})
  lite_cc_library(subgraph_bridge_gather_op_mlu SRCS gather_op.cc DEPS ${subgraph_bridge_deps_mlu})
  lite_cc_library(subgraph_bridge_norm_op_mlu SRCS norm_op.cc DEPS ${subgraph_bridge_deps_mlu})
  set(mlu_subgraph_bridges
      "${mlu_subgraph_bridges}"
      subgraph_bridge_lrn_op_mlu
      subgraph_bridge_gather_op_mlu
      subgraph_bridge_norm_op_mlu
      CACHE INTERNAL "mlu_subgraph_bridges")
endif()

lite_cc_library(subgraph_test_helper_mlu SRCS test_helper.cc DEPS ${mlu_subgraph_bridges})
lite_cc_test(test_conv_converter_mlu SRCS conv_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_act_converter_mlu SRCS act_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_batch_norm_converter_mlu SRCS batch_norm_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_elementwise_converter_mlu SRCS elementwise_ops_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_pool_converter_mlu SRCS pool_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_softmax_converter_mlu SRCS softmax_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_fc_converter_mlu SRCS fc_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_scale_converter_mlu SRCS scale_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_interp_converter_mlu SRCS interpolate_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_concat_converter_mlu SRCS concat_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_transpose_converter_mlu SRCS transpose_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_dropout_converter_mlu SRCS dropout_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_slice_converter_mlu SRCS slice_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_split_converter_mlu SRCS split_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_layout_converter_mlu SRCS layout_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_cast_converter_mlu SRCS cast_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_argmax_converter_mlu SRCS argmax_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_squeeze_converter_mlu SRCS squeeze_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_reshape_converter_mlu SRCS reshape_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
lite_cc_test(test_flatten_converter_mlu SRCS flatten_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)

if (LITE_BUILD_EXTRA)
  lite_cc_test(test_norm_converter_mlu SRCS norm_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
  lite_cc_test(test_lrn_converter_mlu SRCS lrn_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
  lite_cc_test(test_gather_converter_mlu SRCS gather_op_test.cc DEPS core target_wrapper_host model_parser program ${mlu_subgraph_bridges} subgraph_compute_mlu subgraph_test_helper_mlu)
endif()

message(STATUS "+++++ mlu_subgraph_bridges: ${mlu_subgraph_bridges}")
