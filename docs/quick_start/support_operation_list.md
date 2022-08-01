# 支持算子

当前 Paddle Lite 共计支持算子 278 个，其中基础算子 91 个，附加算子 187 个。

### 基础算子

默认编译的算子，共计 91 个。

Host 端 Kernel 是算子在任意 CPU 上纯 C/C++ 的具体实现，具有可移植性强的特点，因此，它一般作为各特定平台算子实现的补充。

以 ARM CPU 为例，如果模型中某个算子没有 ARM 端 Kernel，但是有 Host 端 Kernel，那么模型优化阶段该算子会选择 Host 端 Kernel，该模型还是可以顺利部署。

| OP_name| ARM | OpenCL | Metal | 昆仑芯XPU | Host | X86 | 比特大陆 | 英特尔FPGA | 寒武纪mlu | 华为昇腾NPU | 联发科APU | 瑞芯微NPU | 华为麒麟NPU | 颖脉NNA | 晶晨NPU | 芯原TIM-VX | Android NNAPI| 英特尔OpenVINO |
|-:|-| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -|
|                   affine_channel|Y| | | | | | | | | | | | | | | | | |
|                      affine_grid|Y| | | | | | | | | | | | | | | | | |
|                          arg_max|Y|Y|Y|Y|Y| | | |Y|Y| | |Y| | | | | |
|                     assign_value| | | |Y|Y| |Y| | | | | | | | | | | |
|                       batch_norm|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y| |Y|
|                  bilinear_interp|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | |Y| | |
|               bilinear_interp_v2|Y|Y|Y|Y| |Y| | | |Y| | |Y| | |Y| | |
|                        box_coder|Y|Y|Y|Y|Y|Y|Y| | | | | | | | | | | |
|                            calib|Y| | |Y| |Y| | |Y|Y| | | | | | | | |
|                             cast| | |Y|Y|Y| |Y| |Y|Y| | |Y| | | | | |
|                           concat|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                           conv2d|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                 conv2d_transpose|Y|Y|Y|Y| |Y|Y| | |Y|Y| |Y| |Y|Y|Y| |
|                density_prior_box| | | |Y|Y|Y|Y| | | | | | | | | | | |
|                 depthwise_conv2d|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|       depthwise_conv2d_transpose|Y|Y| | | | |Y| | | | | | | | | | | |
|                          dropout|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y|Y| |
|                  elementwise_add|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                  elementwise_div|Y|Y|Y|Y| |Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y| |
|             elementwise_floordiv|Y|Y| | | |Y| | | | | | | | | | | | |
|                  elementwise_max|Y|Y| |Y| |Y| | | |Y| | |Y|Y| |Y|Y| |
|                  elementwise_min|Y|Y| | | |Y| | | |Y| | |Y|Y| |Y|Y| |
|                  elementwise_mod|Y|Y| | | |Y| | | | | | | | | | | | |
|                  elementwise_mul|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                  elementwise_pow|Y|Y| | | |Y| | | |Y| | |Y| | |Y| | |
|                  elementwise_sub|Y|Y|Y|Y| |Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y| |
|                              elu|Y| | | |Y| | | | | | | | | | | | | |
|                              erf|Y| | | | | | | | | | | | | | | | | |
|                           expand| |Y| | |Y| | | | | | | | | | | | | |
|                        expand_as| | | | |Y| | | | | | | | | | | | | |
|                               fc|Y|Y|Y| | |Y| | |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                             feed| | |Y| |Y| | | | | | | | | | | | | |
|                            fetch| | |Y| |Y| | | | | | | | | | | | | |
|                    fill_constant| | | |Y|Y| |Y| | |Y| | | | | | | | |
|    fill_constant_batch_size_like| | | |Y|Y| | | | |Y| | | | | |Y| | |
|                          flatten| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                         flatten2| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|         flatten_contiguous_range| | | |Y|Y| | | | |Y|Y|Y|Y| |Y|Y| | |
|fusion_elementwise_add_activation|Y|Y|Y| | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|fusion_elementwise_div_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|fusion_elementwise_max_activation|Y| | | | |Y| | | |Y| | |Y|Y| | | | |
|fusion_elementwise_min_activation|Y| | | | |Y| | | |Y| | |Y|Y| | | | |
|fusion_elementwise_mul_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|fusion_elementwise_pow_activation| | | | | | | | | |Y| | |Y| | | | | |
|fusion_elementwise_sub_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|                     grid_sampler|Y|Y| |Y| |Y| | | | | | | | | | | | |
|                    instance_norm|Y|Y| |Y| |Y| | | |Y| | | | | | | | |
|                          io_copy| |Y|Y|Y| | | | |Y| | | | | | | | | |
|                     io_copy_once| |Y|Y|Y| | | | | | | | | | | | | | |
|                           layout|Y|Y| | | |Y| | |Y| | | | | | | | | |
|                      layout_once|Y|Y| | | | | | | | | | | | | | | | |
|                       leaky_relu|Y|Y|Y|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | |
|                 lod_array_length| | | |Y|Y| | | | | | | | | | | | | |
|                           matmul|Y|Y|Y|Y| |Y|Y| | |Y| | |Y|Y| |Y| | |
|                              mul|Y| | |Y| |Y|Y| | |Y| | | | | | | | |
|                   multiclass_nms| | | | |Y| |Y| | | | | | | | | | | |
|                  multiclass_nms2| | | | |Y| |Y| | | | | | | | | | | |
|                  multiclass_nms3| | | | |Y| | | | | | | | | | | | | |
|                   nearest_interp|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y| | |
|                nearest_interp_v2|Y|Y|Y|Y| |Y| | | |Y| | |Y| | |Y| | |
|                            pad2d|Y|Y|Y|Y|Y| | | | |Y| | |Y| | | | | |
|                           pool2d|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                            prelu|Y|Y| |Y|Y| | | | |Y| | |Y| | | | | |
|                        prior_box|Y| | |Y|Y| |Y| | | | | | | | | | | |
|                            range| | | | |Y| | | | |Y| | | | | | | | |
|                      reduce_mean|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | | | | |
|                             relu|Y|Y|Y|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                            relu6|Y|Y|Y|Y|Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                          reshape| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                         reshape2| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                            scale|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                        search_fc| | | |Y| | | | | | | | | | | | | | |
|        sequence_topk_avg_pooling| | | |Y| |Y| | | | | | | | | | | | |
|                  shuffle_channel|Y|Y|Y| |Y| | | | |Y| | |Y| | |Y| | |
|                          sigmoid|Y|Y|Y|Y|Y|Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                            slice|Y|Y|Y|Y| |Y|Y| |Y|Y| | | | | |Y| | |
|                          softmax|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                         softplus|Y| | | |Y| | | | |Y| | |Y| | | | | |
|                          squeeze| |Y| |Y|Y| |Y| |Y|Y| | |Y| | |Y|Y| |
|                         squeeze2| |Y| |Y|Y| |Y| |Y|Y| | |Y| | |Y|Y| |
|                            stack| | | |Y|Y|Y| | | |Y| | | | | | | | |
|                         subgraph| | | | | | | | |Y| | | | | | | | | |
|                  sync_batch_norm|Y|Y| | | |Y| | | | | | | | | | | | |
|                             tanh|Y|Y| |Y|Y|Y| | | |Y|Y|Y|Y| |Y| |Y| |
|                 thresholded_relu|Y| | | |Y| | | | | | | | | | | | | |
|                        transpose|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                       transpose2|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                        unsqueeze| |Y| |Y|Y| | | | |Y| | | | | |Y|Y| |
|                       unsqueeze2| |Y| |Y|Y| | | | |Y| | | | | |Y|Y| |
|                       write_back| | | | |Y| | | | | | | | | | | | | |
|                         yolo_box|Y|Y|Y|Y|Y| |Y| | | | | | | | | | | |


### 附加算子

加上附加算子共计 278 个，需要在编译时打开 `--with_extra=ON` 开关才会编译，具体请参考[参数详情](../source_compile/compile_options)。


| OP_name| ARM | OpenCL | Metal | 昆仑芯XPU | Host | X86 | 比特大陆 | 英特尔FPGA | 寒武纪mlu | 华为昇腾NPU | 联发科APU | 瑞芯微NPU | 华为麒麟NPU | 颖脉NNA | 晶晨NPU | 芯原TIM-VX | Android NNAPI | 英特尔OpenVINO |
|-:|-| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -| -|
|                                            abs|Y|Y| |Y|Y| | | | |Y| | |Y| | | | | |
|                                           acos| |Y| | | | | | | | | | | | | | | | |
|                                 affine_channel|Y| | | | | | | | | | | | | | | | | |
|                                    affine_grid|Y| | | | | | | | | | | | | | | | | |
|                               anchor_generator| | | |Y|Y| | | | | | | | | | | | | |
|                                        arg_max|Y|Y|Y|Y|Y| | | |Y|Y| | |Y| | | | | |
|                                        arg_min| | | | | | | | | |Y| | | | | | | | |
|                                        argsort| | | | |Y| | | | | | | | | | | | | |
|                                           asin| |Y| | | | | | | | | | | | | | | | |
|                                         assign| | | |Y|Y| | | | |Y| | | | | | | | |
|                                   assign_value| | | |Y|Y| |Y| | | | | | | | | | | |
|                                           atan| |Y| | | | | | | | | | | | | | | | |
|                         attention_padding_mask| | | | | | | | | | | | | | | | | | |
|                                           axpy|Y| | | | | | | | | | | | | | | | | |
|                                     batch_norm|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y| |Y|
|                                    beam_search| | | | |Y| | | | | | | | | | | | | |
|                             beam_search_decode| | | | |Y| | | | | | | | | | | | | |
|                                bilinear_interp|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | |Y| | |
|                             bilinear_interp_v2|Y|Y|Y|Y| |Y| | | |Y| | |Y| | |Y| | |
|                                       box_clip| | | |Y|Y| | | | | | | | | | | | | |
|                                      box_coder|Y|Y|Y|Y|Y|Y|Y| | | | | | | | | | | |
|                                          calib|Y| | |Y| |Y| | |Y|Y| | | | | | | | |
|                                     calib_once|Y| | |Y| |Y| | |Y| | | | | | | | | |
|                                           cast| | |Y|Y|Y| |Y| |Y|Y| | |Y| | | | | |
|                                           clip|Y|Y| |Y| |Y| | | |Y| | |Y| | | | | |
|                          collect_fpn_proposals| | | | |Y| | | | | | | | | | | | | |
|                                         concat|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                              conditional_block| | | | |Y| | | | | | | | | | | | | |
|                                         conv2d|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                               conv2d_transpose|Y|Y|Y|Y| |Y|Y| | |Y|Y| |Y| |Y|Y|Y| |
|                                         conv3d| | | |Y| | | | | | | | | | | | | | |
|                                    correlation| | | |Y|Y| | | | | | | | | | | | | |
|                                            cos| |Y| | |Y| | | | | | | | | | | | | |
|                                        cos_sim| | | | |Y| | | | | | | | | | | | | |
|                                   crf_decoding| | | | |Y| | | | | | | | | | | | | |
|                                           crop| | | | |Y| | | | | | | | | | | | | |
|                                    crop_tensor| | | | |Y| | | | | | | | | | | | | |
|                                      ctc_align| | | | |Y| | | | | | | | | | | | | |
|                                         cumsum| | | | |Y| | | | |Y| | | | | | | | |
|                                  decode_bboxes|Y| | | | | | | | | | | | | | | | | |
|                                deformable_conv|Y| | | |Y| | | | |Y| | | | | | | | |
|                              density_prior_box| | | |Y|Y|Y|Y| | | | | | | | | | | |
|                               depthwise_conv2d|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                     depthwise_conv2d_transpose|Y|Y| | | | |Y| | | | | | | | | | | |
|                              dequantize_linear| | | | | | | | | | | | | | | | | | |
|                       distribute_fpn_proposals| | | | |Y| | | | | | | | | | | | | |
|                                        dropout|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y|Y| |
|                                elementwise_add|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                                elementwise_div|Y|Y|Y|Y| |Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y| |
|                           elementwise_floordiv|Y|Y| | | |Y| | | | | | | | | | | | |
|                                elementwise_max|Y|Y| |Y| |Y| | | |Y| | |Y|Y| |Y|Y| |
|                                elementwise_min|Y|Y| | | |Y| | | |Y| | |Y|Y| |Y|Y| |
|                                elementwise_mod|Y|Y| | | |Y| | | | | | | | | | | | |
|                                elementwise_mul|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                                elementwise_pow|Y|Y| | | |Y| | | |Y| | |Y| | |Y| | |
|                                elementwise_sub|Y|Y|Y|Y| |Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y| |
|                                            elu|Y| | | |Y| | | | | | | | | | | | | |
|                                          equal| | |Y| |Y| | | | |Y| | |Y| | | | | |
|                                            erf|Y| | | | | | | | | | | | | | | | | |
|                                            exp|Y|Y|Y|Y|Y| | | | |Y| | |Y| | | | | |
|                                         expand| |Y| | |Y| | | | | | | | | | | | | |
|                                      expand_as| | | | |Y| | | | | | | | | | | | | |
|                                      expand_v2| | | |Y|Y| | | | |Y| | | | | | | | |
|           fake_channel_wise_dequantize_max_abs| | | | | | | | | | | | | | | | | | |
|  fake_channel_wise_quantize_dequantize_abs_max| | | | | | | | | | | | | | | | | | |
|                        fake_dequantize_max_abs| | | | | | | | | | | | | | | | | | |
|                          fake_quantize_abs_max| | | | | | | | | | | | | | | | | | |
|               fake_quantize_dequantize_abs_max| | | | | | | | | | | | | | | | | | |
|fake_quantize_dequantize_moving_average_abs_max| | | | | | | | | | | | | | | | | | |
|           fake_quantize_moving_average_abs_max| | | | | | | | | | | | | | | | | | |
|                    fake_quantize_range_abs_max| | | | | | | | | | | | | | | | | | |
|                                             fc|Y|Y|Y| | |Y| | |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                                           feed| | |Y| |Y| | | | | | | | | | | | | |
|                                          fetch| | |Y| |Y| | | | | | | | | | | | | |
|                                  fill_any_like| | | |Y|Y| | | | |Y| | | | | |Y| | |
|                                  fill_constant| | | |Y|Y| |Y| | |Y| | | | | | | | |
|                  fill_constant_batch_size_like| | | |Y|Y| | | | |Y| | | | | |Y| | |
|                                fill_zeros_like| | | |Y|Y| | | | | | | | | | | | | |
|                                        flatten| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                                       flatten2| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                       flatten_contiguous_range| | | |Y|Y| | | | |Y|Y|Y|Y| |Y|Y| | |
|                                           flip| | | | |Y| | | | | | | | | | | | | |
|                                          floor|Y| | | |Y| | | | |Y| | |Y| | | | | |
|              fusion_elementwise_add_activation|Y|Y|Y| | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|              fusion_elementwise_div_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|              fusion_elementwise_max_activation|Y| | | | |Y| | | |Y| | |Y|Y| | | | |
|              fusion_elementwise_min_activation|Y| | | | |Y| | | |Y| | |Y|Y| | | | |
|              fusion_elementwise_mul_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|              fusion_elementwise_pow_activation| | | | | | | | | |Y| | |Y| | | | | |
|              fusion_elementwise_sub_activation|Y|Y| | | |Y| | | |Y|Y|Y|Y|Y|Y| | | |
|                                         gather| |Y| |Y|Y|Y| | |Y|Y| | |Y| | | | | |
|                                      gather_nd| | | | |Y| | | | | | | | | | | | | |
|                                    gather_tree| | | | |Y| | | | | | | | | | | | | |
|                                gaussian_random| | | | |Y| | | | | | | | | | | | | |
|                                           gelu|Y|Y| |Y| |Y| | | |Y| | |Y| | | | | |
|                             generate_proposals| | | |Y|Y| | | | | | | | | | | | | |
|                          generate_proposals_v2| | | | |Y| | | | | | | | | | | | | |
|                                  greater_equal| | | | |Y| | | | |Y| | |Y| | | | | |
|                                   greater_than| |Y| | |Y| | | | |Y| | |Y| | | | | |
|                                   grid_sampler|Y|Y| |Y| |Y| | | | | | | | | | | | |
|                                     group_norm|Y| | | | |Y| | | |Y| | | | | | | | |
|                                            gru|Y| | |Y| |Y| | | | | | | | | | | | |
|                                       gru_unit|Y| | |Y| |Y| | | | | | | | | | | | |
|                                   hard_sigmoid|Y|Y|Y|Y|Y| |Y| | |Y| | |Y| | |Y| | |
|                                     hard_swish|Y|Y|Y|Y|Y|Y|Y| | |Y| | |Y| | |Y| | |
|                                    im2sequence| | | |Y|Y| |Y| | | | | | | | | | | |
|                                      increment| | | |Y|Y| | | | | | | | | | | | | |
|                                   index_select| | | | |Y| | | | | | | | | | | | | |
|                                  instance_norm|Y|Y| |Y| |Y| | | |Y| | | | | | | | |
|                                        inverse| | | | |Y| | | | | | | | | | | | | |
|                                        io_copy| |Y|Y|Y| | | | |Y| | | | | | | | | |
|                                   io_copy_once| |Y|Y|Y| | | | | | | | | | | | | | |
|                                       is_empty| | | |Y|Y| | | | | | | | | | | | | |
|                                     layer_norm|Y|Y| |Y| |Y| | | |Y| | |Y| | | | | |
|                                         layout|Y|Y| | | |Y| | |Y| | | | | | | | | |
|                                    layout_once|Y|Y| | | | | | | | | | | | | | | | |
|                                     leaky_relu|Y|Y|Y|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | |
|                                     less_equal| | | | |Y| | | | |Y| | |Y| | | | | |
|                                      less_than| | | |Y|Y| | | | |Y| | |Y| | | | | |
|                                       linspace| | | | |Y| | | | | | | | | | | | | |
|                               lod_array_length| | | |Y|Y| | | | | | | | | | | | | |
|                                      lod_reset| | | | |Y| | | | | | | | | | | | | |
|                                            log|Y|Y| |Y|Y| | | | |Y| | |Y| | | | | |
|                                    logical_and| | | |Y|Y| | | | |Y| | |Y| | | | | |
|                                    logical_not| | | |Y|Y| | | | |Y| | |Y| | | | | |
|                                     logical_or| | | | |Y| | | | | | | | | | | | | |
|                                    logical_xor| | | | |Y| | | | | | | | | | | | | |
|                                   lookup_table|Y| | |Y| |Y| | | | | | | | | | | | |
|                           lookup_table_dequant|Y| | | | | | | | | | | | | | | | | |
|                                lookup_table_v2|Y| | |Y| |Y| | | |Y| | |Y| | | | | |
|                                            lrn|Y|Y| |Y| | | | |Y| | | | | | | | | |
|                                           lstm|Y| | | | | | | | | | | | | | | | | |
|                            match_matrix_tensor| | | |Y| |Y| | | | | | | | | | | | |
|                                         matmul|Y|Y|Y|Y| |Y|Y| | |Y| | |Y|Y| |Y| | |
|                                      matmul_v2|Y|Y|Y|Y| | | | | |Y| | |Y|Y| |Y| | |
|                                     matrix_nms| | | | |Y| | | | | | | | | | | | | |
|                          max_pool2d_with_index| | | |Y|Y| |Y| | | | | | | | | | | |
|                                           mean|Y| | | | | | | | | | | | | | | | | |
|                               merge_lod_tensor| | | | |Y| | | | | | | | | | | | | |
|                                       meshgrid| | | | |Y| | | | |Y| | | | | | | | |
|                                           mish|Y| | | | |Y| | | | | | | | | | | | |
|                                            mul|Y| | |Y| |Y|Y| | |Y| | | | | | | | |
|                                 multiclass_nms| | | | |Y| |Y| | | | | | | | | | | |
|                                multiclass_nms2| | | | |Y| |Y| | | | | | | | | | | |
|                                multiclass_nms3| | | | |Y| | | | | | | | | | | | | |
|                                 nearest_interp|Y|Y|Y|Y| |Y|Y| |Y|Y| | |Y| | |Y| | |
|                              nearest_interp_v2|Y|Y|Y|Y| |Y| | | |Y| | |Y| | |Y| | |
|                                       negative|Y| | | | | | | | | | | | | | | | | |
|                                           norm| | | |Y|Y| |Y| |Y|Y| | |Y| | | | | |
|                                      not_equal| | | | |Y| | | | |Y| | |Y| | | | | |
|                                        one_hot| | | | |Y| | | | | | | | | | | | | |
|                                     one_hot_v2| | | | |Y| | | | | | | | | | | | | |
|                                         p_norm| | | | |Y| | | | | | | | | | | | | |
|                                          pad2d|Y|Y|Y|Y|Y| | | | |Y| | |Y| | | | | |
|                                          pad3d| | | |Y|Y| | | | |Y| | | | | | | | |
|                                  pixel_shuffle|Y|Y| |Y|Y| | | | | | | | | | | | | |
|                          polygon_box_transform| | | | |Y| | | | | | | | | | | | | |
|                                         pool2d|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                                            pow|Y| | |Y| |Y| | | |Y| | |Y| | | | | |
|                                          prelu|Y|Y| |Y|Y| | | | |Y| | |Y| | | | | |
|                                          print| | | | |Y| | | | | | | | | | | | | |
|                                      prior_box|Y| | |Y|Y| |Y| | | | | | | | | | | |
|                                quantize_linear| | | | | | | | | | | | | | | | | | |
|                                          range| | | | |Y| | | | |Y| | | | | | | | |
|                                read_from_array| | | |Y|Y| | | | | | | | | | | | | |
|                                     reciprocal|Y| | |Y|Y| | | | | | | | | | | | | |
|                                     reduce_all| | | |Y|Y| | | | | | | | | | | | | |
|                                     reduce_any| | | |Y|Y| | | | | | | | | | | | | |
|                                     reduce_max|Y|Y|Y|Y| |Y|Y| | | | | | | | | | | |
|                                    reduce_mean|Y|Y|Y|Y| |Y|Y| | |Y| | |Y| | | | | |
|                                     reduce_min|Y| |Y|Y| |Y| | | | | | | | | | | | |
|                                    reduce_prod|Y| | |Y| |Y| | | | | | | | | | | | |
|                                     reduce_sum|Y| |Y|Y| |Y|Y| | |Y| | |Y| | | | | |
|                                           relu|Y|Y|Y|Y|Y|Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                                          relu6|Y|Y|Y|Y|Y|Y| | |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                                   relu_clipped|Y| | | |Y| | | | | | | | | | | | | |
|                                        reshape| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                                       reshape2| |Y|Y|Y|Y| |Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y| |
|                     retinanet_detection_output| | | | |Y| | | | | | | | | | | | | |
|                                        reverse| | | | |Y| | | | | | | | | | | | | |
|                                            rnn|Y| | |Y| |Y| | | | | | | | | | | | |
|                                      roi_align| | | |Y|Y| | | | | | | | | | | | | |
|                      roi_perspective_transform| | | | |Y| | | | | | | | | | | | | |
|                                          rsqrt|Y|Y| |Y|Y|Y| | | | | | | | | | | | |
|                                    sampling_id| | | | |Y| | | | | | | | | | | | | |
|                                          scale|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                                        scatter|Y| | | | | | | | | | | | | | | | | |
|                                 scatter_nd_add| | | | |Y| | | | | | | | | | | | | |
|                         search_aligned_mat_mul| | | | | |Y| | | | | | | | | | | | |
|                  search_attention_padding_mask| | | | | |Y| | | | | | | | | | | | |
|                                      search_fc| | | |Y| | | | | | | | | | | | | | |
|                                    search_grnn| | | |Y| |Y| | | | | | | | | | | | |
|                           search_group_padding| | | | | |Y| | | | | | | | | | | | |
|                          search_seq_arithmetic| | | |Y| |Y| | | | | | | | | | | | |
|                           search_seq_depadding| | | | | |Y| | | | | | | | | | | | |
|                                  search_seq_fc| | | | | |Y| | | | | | | | | | | | |
|                             search_seq_softmax| | | | | |Y| | | | | | | | | | | | |
|                                   select_input| | | | |Y| | | | | | | | | | | | | |
|                            sequence_arithmetic| | | |Y| |Y| | | | | | | | | | | | |
|                                sequence_concat| | | |Y| |Y| | | | | | | | | | | | |
|                                  sequence_conv|Y| | | | |Y| | | | | | | | | | | | |
|                                sequence_expand| | | | |Y| | | | | | | | | | | | | |
|                             sequence_expand_as|Y| | | | |Y| | | | | | | | | | | | |
|                                  sequence_mask| | | |Y|Y| | | | | | | | | | | | | |
|                                   sequence_pad| | | |Y|Y| | | | | | | | | | | | | |
|                                  sequence_pool|Y| | |Y| |Y| | | | | | | | | | | | |
|                               sequence_reshape| | | | | |Y| | | | | | | | | | | | |
|                               sequence_reverse| | | |Y| |Y| | | | | | | | | | | | |
|                               sequence_softmax| | | | |Y| | | | | | | | | | | | | |
|                      sequence_topk_avg_pooling| | | |Y| |Y| | | | | | | | | | | | |
|                                 sequence_unpad| | | |Y|Y| | | | | | | | | | | | | |
|                                          shape| |Y| |Y|Y| |Y| | |Y| | | | | | | | |
|                                shuffle_channel|Y|Y|Y| |Y| | | | |Y| | |Y| | |Y| | |
|                                        sigmoid|Y|Y|Y|Y|Y|Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                                           sign|Y| | |Y| | | | | | | | | | | | | | |
|                                            sin| |Y| | |Y| | | | | | | | | | | | | |
|                                          slice|Y|Y|Y|Y| |Y|Y| |Y|Y| | | | | |Y| | |
|                                        softmax|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y|Y|Y|Y|Y|Y|
|                                       softplus|Y| | | |Y| | | | |Y| | |Y| | | | | |
|                                       softsign| | | |Y| |Y| | | | | | | | | | | | |
|                                  sparse_conv2d|Y| | | | | | | | | | | | | | | | | |
|                                          split| |Y|Y|Y|Y| |Y| |Y|Y| | |Y| | |Y| | |
|                               split_lod_tensor|Y| | | | | | | | | | | | | | | | | |
|                                           sqrt|Y|Y| |Y| |Y|Y| | | | | | | | | | | |
|                                         square|Y|Y| |Y|Y|Y|Y| | |Y| | |Y| | | | | |
|                                        squeeze| |Y| |Y|Y| |Y| |Y|Y| | |Y| | |Y|Y| |
|                                       squeeze2| |Y| |Y|Y| |Y| |Y|Y| | |Y| | |Y|Y| |
|                                          stack| | | |Y|Y|Y| | | |Y| | | | | | | | |
|                                  strided_slice| | | | |Y| | | | |Y| | |Y| | | | | |
|                                       subgraph| | | | | | | | |Y| | | | | | | | | |
|                                            sum|Y| | |Y| | | | | |Y| | | | | | | | |
|                                          swish|Y|Y|Y|Y|Y| |Y| | |Y| | |Y| | | | | |
|                                sync_batch_norm|Y|Y| | | |Y| | | | | | | | | | | | |
|                                            tan| |Y| | | | | | | | | | | | | | | | |
|                                           tanh|Y|Y| |Y|Y|Y| | | |Y|Y|Y|Y| |Y| |Y| |
|                         tensor_array_to_tensor| | | | |Y| | | | | | | | | | | | | |
|                               thresholded_relu|Y| | | |Y| | | | | | | | | | | | | |
|                                           tile| | | | |Y| | | | |Y| | |Y| | | | | |
|                                          top_k| | | |Y|Y| | | | |Y| | | | | | | | |
|                                       top_k_v2| | | | |Y| | | | |Y| | | | | | | | |
|                                      transpose|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                                     transpose2|Y|Y|Y|Y| |Y|Y| |Y|Y|Y|Y|Y| |Y|Y|Y| |
|                                      tril_triu| | | | |Y| | | | | | | | | | | | | |
|                                         unbind| | | | |Y| | | | | | | | | | | | | |
|                                         unfold| | | | |Y| | | | | | | | | | | | | |
|                                 uniform_random| | | | |Y| | | | | | | | | | | | | |
|                             unique_with_counts| | | | |Y| | | | | | | | | | | | | |
|                                      unsqueeze| |Y| |Y|Y| | | | |Y| | | | | |Y|Y| |
|                                     unsqueeze2| |Y| |Y|Y| | | | |Y| | | | | |Y|Y| |
|                                        unstack| | | |Y|Y| | | | | | | | | | | | | |
|                                    var_conv_2d| | | |Y| |Y| | | | | | | | | | | | |
|                                          where| | | | |Y| | | | |Y| | | | | | | | |
|                                    where_index| | | | |Y| | | | | | | | | | | | | |
|                                          while| | | | |Y| | | | | | | | | | | | | |
|                                     write_back| | | | |Y| | | | | | | | | | | | | |
|                                 write_to_array| | | |Y|Y| | | | | | | | | | | | | |
|                                       yolo_box|Y|Y|Y|Y|Y| |Y| | | | | | | | | | | |
|                                   __xpu__bigru| | | |Y| | | | | | | | | | | | | | |
|                                  __xpu__conv2d| | | |Y| | | | | | | | | | | | | | |
|                    __xpu__dynamic_lstm_fuse_op| | | |Y| | | | | | | | | | | | | | |
|              __xpu__embedding_with_eltwise_add| | | |Y| | | | | | | | | | | | | | |
|                                      __xpu__fc| | | |Y| | | | | | | | | | | | | | |
|                       __xpu__generate_sequence| | | |Y| | | | | | | | | | | | | | |
|                                   __xpu__logit| | | |Y| | | | | | | | | | | | | | |
|                       __xpu__mmdnn_bid_emb_att| | | |Y| | | | | | | | | | | | | | |
|                  __xpu__mmdnn_bid_emb_grnn_att| | | |Y| | | | | | | | | | | | | | |
|                 __xpu__mmdnn_bid_emb_grnn_att2| | | |Y| | | | | | | | | | | | | | |
|                   __xpu__mmdnn_match_conv_topk| | | |Y| | | | | | | | | | | | | | |
|                         __xpu__mmdnn_merge_all| | | |Y| | | | | | | | | | | | | | |
|                  __xpu__mmdnn_search_attention| | | |Y| | | | | | | | | | | | | | |
|                 __xpu__mmdnn_search_attention2| | | |Y| | | | | | | | | | | | | | |
|                           __xpu__multi_encoder| | | |Y| | | | | | | | | | | | | | |
|                           __xpu__multi_softmax| | | |Y| | | | | | | | | | | | | | |
|                                __xpu__resnet50| | | |Y| | | | | | | | | | | | | | |
|                             __xpu__resnet_cbam| | | |Y| | | | | | | | | | | | | | |
|                                __xpu__sfa_head| | | |Y| | | | | | | | | | | | | | |
|                            __xpu__softmax_topk| | | |Y| | | | | | | | | | | | | | |
|                __xpu__squeeze_excitation_block| | | |Y| | | | | | | | | | | | | | |
