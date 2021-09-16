# 支持算子

当前Paddle-Lite共计支持算子209个，其中基础算子78个，附加算子131个。

### 基础算子

默认编译的算子，共计78个。

Host端Kernel是算子在任意CPU上纯C/C++的具体实现，具有可移植性强的特点，因此，它一般作为各特定平台算子实现的补充。

举例PaddleLite在ARM上部署模型，如果模型中某个算子没有ARM端Kernel，但是有Host端Kernel，那么模型优化阶段该算子会选择Host端Kernel，该模型还是可以顺利部署。

| OP Name | Host | X86 | CUDA | ARM | OpenCL | FPGA | 华为NPU | 百度XPU | 瑞芯微NPU | 联发科APU | 颖脉NNA | 英特尔FPGA |
|-:|-|-|-|-|-|-|-|-|-|-|-|-|
| affine_channel | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| affine_grid | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| arg_max | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| assign_value | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| batch_norm | 　 | Y | 　 | Y | 　 | 　 | Y | Y | Y | 　 |　 |　 |
| bilinear_interp | 　 | 　 | Y | Y | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| box_coder | 　 | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 |　 |　 |
| calib | 　 | 　 | Y | Y | 　 | Y | 　 | 　 | 　 | 　 |　 |　 |
| cast | 　 | Y | 　 | Y | 　 | 　 | 　 | Y | 　 | 　 |　 |　 |
| concat | 　 | Y | Y | Y | Y | 　 | Y | 　 | Y | Y |　|　 |
| conv2d | 　 | Y | Y | Y | Y | Y | Y | Y | Y | Y |　Y | Y |
| conv2d_transpose | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | Y |　 |　 |
| density_prior_box | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| depthwise_conv2d | 　 | Y | Y | Y | Y | Y | Y | Y | Y | Y |　Y | Y |
| depthwise_conv2d_transpose | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| dropout | 　 | Y | Y | Y | Y | Y | Y | Y | 　 | 　 |　 |　 |
| elementwise_add | 　 | Y | Y | Y | Y | Y | Y | Y | Y | Y |　 |　 |
| elementwise_div | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | Y | 　 |　 |　 |
| elementwise_max | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| elementwise_mod | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| elementwise_mul | 　 | Y | Y | Y | Y | Y | Y | 　 | Y | Y |　 |　 |
| elementwise_pow | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| elementwise_sub | 　 | Y | Y | Y | Y | 　 | Y | 　 | Y | 　 |　 |　 |
| elu | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| expand | Y | 　 | 　 | 　 | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| expand_as | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| fc | 　 | Y | Y | Y | Y | Y | Y | 　 | Y | Y |　Y |　 |
| feed | Y | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 | 　 |　 |　 |
| fetch | Y | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 |　 |　 |
| fill_constant | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| fill_constant_batch_size_like | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| flatten | Y | 　 | 　 | 　 | Y | 　 | 　 | 　 | Y | 　 |　 |　 |
| flatten2 | Y | 　 | 　 | 　 | Y | 　 | 　 | 　 | Y | 　 |　 |　 |
| fusion_elementwise_add_activation | 　 | 　 | Y | Y | Y | Y | Y | 　 | 　 | Y  |　 |　 |
| fusion_elementwise_div_activation | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| fusion_elementwise_max_activation | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| fusion_elementwise_mul_activation | 　 | 　 | Y | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| fusion_elementwise_sub_activation | 　 | 　 | Y | Y | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| grid_sampler | 　 | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 |　 |　 |
| instance_norm | 　 | 　 | 　 | Y | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| io_copy | 　 | 　 | Y | 　 | Y | Y | 　 | 　 | 　 | 　 |　 |　 |
| io_copy_once | 　 | 　 | Y | 　 | Y | Y | 　 | 　 | 　 | 　 |　 |　 |
| layout | 　 | 　 | Y | Y | Y | Y | 　 | 　 | 　 | 　 |　 |　 |
| leaky_relu | 　 | Y | Y | Y | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| matmul | 　 | Y | Y | Y | 　 | 　 | Y | Y | 　 | 　 |　 |　 |
| mul | 　 | Y | Y | Y | 　 | 　 | Y | Y | 　 | 　 |　 |　 |
| multiclass_nms | Y | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 |　 |　 |
| multiclass_nms2 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| nearest_interp | 　 | 　 | Y | Y | Y | 　 | Y | 　 | 　 | 　 |　 |　 |
| pad2d | 　 | 　 | 　 | Y | Y | 　 | Y | 　 | Y | 　 |　 |　 |
| pool2d | 　 | Y | Y | Y | Y | Y | Y | Y | Y | Y |　Y |　 |
| prelu | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| prior_box | 　 | 　 | 　 | Y | 　 | Y | 　 | 　 | 　 | 　 |　 |　 |
| range | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| reduce_mean | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| relu | 　 | Y | Y | Y | Y | 　 | Y | 　 | Y | Y |　Y |　 |
| relu6 | 　 | 　 | 　 | Y | Y | 　 | Y | 　 | Y | 　 |　 |　 |
| reshape | Y | Y | 　 | 　 | Y | 　 | Y | Y | 　 | 　 |　 |　 |
| reshape2 | Y | Y | 　 | 　 | Y | 　 | Y | Y | Y | 　 |　 |　 |
| scale | 　 | Y | Y | Y | Y | Y | Y | Y | Y | 　 |　 |　 |
| search_fc | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| sequence_topk_avg_pooling | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| shuffle_channel | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| sigmoid | 　 | Y | Y | Y | Y | 　 | Y | 　 | Y | 　 |　 |　 |
| slice | 　 | Y | 　 | Y | Y | 　 | 　 | Y | 　 | 　 |　 |　 |
| softmax | 　 | Y | Y | Y | 　 | 　 | Y | Y | Y | Y |　 |　 |
| split | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| squeeze | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| squeeze2 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |　 |　 |
| stack | 　 | Y | 　 | Y | 　 | 　 | 　 | Y | 　 | 　 |　 |　 |
| subgraph | 　 | 　 | 　 | 　 | 　 | 　 | Y | Y | Y | Y |　 |　 |
| tanh | 　 | Y | Y | Y | Y | 　 | Y | Y | 　 | 　 |　 |　 |
| thresholded_relu | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| transpose | 　 | Y | Y | Y | Y | 　 | Y | Y | 　 | 　 |　 |　 |
| transpose2 | 　 | Y | Y | Y | Y | 　 | Y | Y | Y | 　 |　 |　 |
| unsqueeze | Y | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| unsqueeze2 | Y | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |　 |　 |
| yolo_box | 　 | 　 | Y | Y | 　 | 　 | 　 | Y | 　 | 　 |　 |　 |


### 附加算子

附加算子共计131个，需要在编译时打开`--with_extra=ON`开关才会编译，具体请参考[参数详情](../source_compile/library)。

| OP Name | Host | X86 | CUDA | ARM | OpenCL | FPGA | 华为NPU | 百度XPU | 瑞芯微NPU | 联发科APU | 英特尔FPGA |
|-:|-|-|-|-|-|-|-|-|-|-|-|
| abs | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| anchor_generator | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| assign | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| attention_padding_mask | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| axpy | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| beam_search_decode | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| beam_search_decode | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| box_clip | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| calib_once | 　 | 　 | Y | Y | 　 | Y | 　 | 　 | 　 | 　 | 　 |
| clip | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| collect_fpn_proposals | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| conditional_block | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| correlation | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| crf_decoding | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| crop | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| ctc_align | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| decode_bboxes | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| deformable_conv | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| distribute_fpn_proposals | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| equal | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| exp | 　 | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_channel_wise_dequantize_max_abs | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_dequantize_max_abs | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_quantize_abs_max | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_quantize_dequantize_abs_max | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_quantize_dequantize_moving_average_abs_max | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_quantize_moving_average_abs_max | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| fake_quantize_range_abs_max | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| floor | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| gather | 　 | Y | 　 | Y | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| gelu | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| generate_proposals | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| greater_equal | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| greater_than | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| group_norm | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| gru | 　 | Y | Y | Y | 　 | Y | 　 | 　 | 　 | 　 | 　 |
| gru_unit | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| hard_sigmoid | 　 | 　 | 　 | Y | Y | 　 | Y | 　 | 　 | 　 | 　 |
| hard_swish | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| im2sequence | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| increment | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| is_empty | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| layer_norm | 　 | Y | 　 | Y | 　 | 　 | Y | Y | 　 | 　 | 　 |
| layout_once | 　 | 　 | Y | Y | 　 | Y | 　 | 　 | 　 | 　 | 　 |
| less_equal | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| less_than | Y | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| lod_reset | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| log | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| logical_and | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| logical_not | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| logical_or | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| logical_xor | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| lookup_table | 　 | Y | Y | Y | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| lookup_table_dequant | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| lookup_table_v2 | 　 | Y | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| lrn | 　 | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 |
| lstm | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| match_matrix_tensor | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| max_pool2d_with_index | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| mean | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| merge_lod_tensor | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| negative | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| norm | 　 | 　 | 　 | Y | 　 | Y | 　 | 　 | 　 | 　 | 　 |
| not_equal | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| one_hot | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| pixel_shuffle | Y | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 |
| polygon_box_transform | Y | 　 | 　 |   | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| pow | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| print | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| read_from_array | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| reciprocal | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| reduce_max | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| reduce_prod | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| reduce_sum | 　 | Y | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| relu_clipped | 　 | 　 | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| retinanet_detection_output | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| roi_align | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| roi_perspective_transform | Y | 　 | 　 |   | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| rsqrt | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| scatter_nd_add | Y　 |   |   | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_aligned_mat_mul | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_attention_padding_mask | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_grnn | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_group_padding | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_seq_arithmetic | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_seq_depadding | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_seq_fc | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| search_seq_softmax | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_arithmetic | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_concat | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_conv | 　 | Y | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_expand | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_expand_as | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_mask | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_pad | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_pool | 　 | Y | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_pool_concat | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_reshape | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_reverse | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_reverse_embedding | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_softmax | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sequence_unpad |  Y　|  | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| shape | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sign | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| softsign | 　 | Y | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| split_lod_tensor | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| sqrt | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| square | 　 | Y | 　 | Y | 　 | 　 | Y | 　 | 　 | 　 | 　 |
| swish | 　 | 　 | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 |
| top_k | 　 | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| topk_pooling | 　 | 　 | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| uniform_random | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| var_conv_2d | 　 | Y | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| where_index | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| while | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| write_to_array | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| __xpu__conv2d | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__embedding_with_eltwise_add | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__fc | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__mmdnn_bid_emb_att | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__mmdnn_bid_emb_grnn_att | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__mmdnn_bid_emb_grnn_att2 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__mmdnn_match_conv_topk | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__mmdnn_merge_all | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 |
| __xpu__mmdnn_search_attention | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__multi_encoder | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__resnet_cbam | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__resnet50 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| __xpu__sfa_head | 　 | 　 | 　 | 　 | 　 | 　 | 　 | Y | 　 | 　 | 　 |
| matrix_nms | Y | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
