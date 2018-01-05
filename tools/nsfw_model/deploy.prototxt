name: "ResNet_50_1by2_nsfw"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 224 dim: 224 } }
}
layer {
  name: "conv_1"
  type: "Convolution"
  bottom: "data"
  top: "conv_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 3
    kernel_size: 7
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_1"
  type: "BatchNorm"
  bottom: "conv_1"
  top: "conv_1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_1"
  type: "Scale"
  bottom: "conv_1"
  top: "conv_1"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_1"
  type: "ReLU"
  bottom: "conv_1"
  top: "conv_1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv_1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv_stage0_block0_proj_shortcut"
  type: "Convolution"
  bottom: "pool1"
  top: "conv_stage0_block0_proj_shortcut"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block0_proj_shortcut"
  type: "BatchNorm"
  bottom: "conv_stage0_block0_proj_shortcut"
  top: "conv_stage0_block0_proj_shortcut"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block0_proj_shortcut"
  type: "Scale"
  bottom: "conv_stage0_block0_proj_shortcut"
  top: "conv_stage0_block0_proj_shortcut"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv_stage0_block0_branch2a"
  type: "Convolution"
  bottom: "pool1"
  top: "conv_stage0_block0_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block0_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage0_block0_branch2a"
  top: "conv_stage0_block0_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block0_branch2a"
  type: "Scale"
  bottom: "conv_stage0_block0_branch2a"
  top: "conv_stage0_block0_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block0_branch2a"
  type: "ReLU"
  bottom: "conv_stage0_block0_branch2a"
  top: "conv_stage0_block0_branch2a"
}
layer {
  name: "conv_stage0_block0_branch2b"
  type: "Convolution"
  bottom: "conv_stage0_block0_branch2a"
  top: "conv_stage0_block0_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block0_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage0_block0_branch2b"
  top: "conv_stage0_block0_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block0_branch2b"
  type: "Scale"
  bottom: "conv_stage0_block0_branch2b"
  top: "conv_stage0_block0_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block0_branch2b"
  type: "ReLU"
  bottom: "conv_stage0_block0_branch2b"
  top: "conv_stage0_block0_branch2b"
}
layer {
  name: "conv_stage0_block0_branch2c"
  type: "Convolution"
  bottom: "conv_stage0_block0_branch2b"
  top: "conv_stage0_block0_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block0_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage0_block0_branch2c"
  top: "conv_stage0_block0_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block0_branch2c"
  type: "Scale"
  bottom: "conv_stage0_block0_branch2c"
  top: "conv_stage0_block0_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage0_block0"
  type: "Eltwise"
  bottom: "conv_stage0_block0_proj_shortcut"
  bottom: "conv_stage0_block0_branch2c"
  top: "eltwise_stage0_block0"
}
layer {
  name: "relu_stage0_block0"
  type: "ReLU"
  bottom: "eltwise_stage0_block0"
  top: "eltwise_stage0_block0"
}
layer {
  name: "conv_stage0_block1_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage0_block0"
  top: "conv_stage0_block1_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block1_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage0_block1_branch2a"
  top: "conv_stage0_block1_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block1_branch2a"
  type: "Scale"
  bottom: "conv_stage0_block1_branch2a"
  top: "conv_stage0_block1_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block1_branch2a"
  type: "ReLU"
  bottom: "conv_stage0_block1_branch2a"
  top: "conv_stage0_block1_branch2a"
}
layer {
  name: "conv_stage0_block1_branch2b"
  type: "Convolution"
  bottom: "conv_stage0_block1_branch2a"
  top: "conv_stage0_block1_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block1_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage0_block1_branch2b"
  top: "conv_stage0_block1_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block1_branch2b"
  type: "Scale"
  bottom: "conv_stage0_block1_branch2b"
  top: "conv_stage0_block1_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block1_branch2b"
  type: "ReLU"
  bottom: "conv_stage0_block1_branch2b"
  top: "conv_stage0_block1_branch2b"
}
layer {
  name: "conv_stage0_block1_branch2c"
  type: "Convolution"
  bottom: "conv_stage0_block1_branch2b"
  top: "conv_stage0_block1_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block1_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage0_block1_branch2c"
  top: "conv_stage0_block1_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block1_branch2c"
  type: "Scale"
  bottom: "conv_stage0_block1_branch2c"
  top: "conv_stage0_block1_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage0_block1"
  type: "Eltwise"
  bottom: "eltwise_stage0_block0"
  bottom: "conv_stage0_block1_branch2c"
  top: "eltwise_stage0_block1"
}
layer {
  name: "relu_stage0_block1"
  type: "ReLU"
  bottom: "eltwise_stage0_block1"
  top: "eltwise_stage0_block1"
}
layer {
  name: "conv_stage0_block2_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage0_block1"
  top: "conv_stage0_block2_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block2_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage0_block2_branch2a"
  top: "conv_stage0_block2_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block2_branch2a"
  type: "Scale"
  bottom: "conv_stage0_block2_branch2a"
  top: "conv_stage0_block2_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block2_branch2a"
  type: "ReLU"
  bottom: "conv_stage0_block2_branch2a"
  top: "conv_stage0_block2_branch2a"
}
layer {
  name: "conv_stage0_block2_branch2b"
  type: "Convolution"
  bottom: "conv_stage0_block2_branch2a"
  top: "conv_stage0_block2_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block2_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage0_block2_branch2b"
  top: "conv_stage0_block2_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block2_branch2b"
  type: "Scale"
  bottom: "conv_stage0_block2_branch2b"
  top: "conv_stage0_block2_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage0_block2_branch2b"
  type: "ReLU"
  bottom: "conv_stage0_block2_branch2b"
  top: "conv_stage0_block2_branch2b"
}
layer {
  name: "conv_stage0_block2_branch2c"
  type: "Convolution"
  bottom: "conv_stage0_block2_branch2b"
  top: "conv_stage0_block2_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage0_block2_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage0_block2_branch2c"
  top: "conv_stage0_block2_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage0_block2_branch2c"
  type: "Scale"
  bottom: "conv_stage0_block2_branch2c"
  top: "conv_stage0_block2_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage0_block2"
  type: "Eltwise"
  bottom: "eltwise_stage0_block1"
  bottom: "conv_stage0_block2_branch2c"
  top: "eltwise_stage0_block2"
}
layer {
  name: "relu_stage0_block2"
  type: "ReLU"
  bottom: "eltwise_stage0_block2"
  top: "eltwise_stage0_block2"
}
layer {
  name: "conv_stage1_block0_proj_shortcut"
  type: "Convolution"
  bottom: "eltwise_stage0_block2"
  top: "conv_stage1_block0_proj_shortcut"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block0_proj_shortcut"
  type: "BatchNorm"
  bottom: "conv_stage1_block0_proj_shortcut"
  top: "conv_stage1_block0_proj_shortcut"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block0_proj_shortcut"
  type: "Scale"
  bottom: "conv_stage1_block0_proj_shortcut"
  top: "conv_stage1_block0_proj_shortcut"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv_stage1_block0_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage0_block2"
  top: "conv_stage1_block0_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block0_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage1_block0_branch2a"
  top: "conv_stage1_block0_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block0_branch2a"
  type: "Scale"
  bottom: "conv_stage1_block0_branch2a"
  top: "conv_stage1_block0_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block0_branch2a"
  type: "ReLU"
  bottom: "conv_stage1_block0_branch2a"
  top: "conv_stage1_block0_branch2a"
}
layer {
  name: "conv_stage1_block0_branch2b"
  type: "Convolution"
  bottom: "conv_stage1_block0_branch2a"
  top: "conv_stage1_block0_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block0_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage1_block0_branch2b"
  top: "conv_stage1_block0_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block0_branch2b"
  type: "Scale"
  bottom: "conv_stage1_block0_branch2b"
  top: "conv_stage1_block0_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block0_branch2b"
  type: "ReLU"
  bottom: "conv_stage1_block0_branch2b"
  top: "conv_stage1_block0_branch2b"
}
layer {
  name: "conv_stage1_block0_branch2c"
  type: "Convolution"
  bottom: "conv_stage1_block0_branch2b"
  top: "conv_stage1_block0_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block0_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage1_block0_branch2c"
  top: "conv_stage1_block0_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block0_branch2c"
  type: "Scale"
  bottom: "conv_stage1_block0_branch2c"
  top: "conv_stage1_block0_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage1_block0"
  type: "Eltwise"
  bottom: "conv_stage1_block0_proj_shortcut"
  bottom: "conv_stage1_block0_branch2c"
  top: "eltwise_stage1_block0"
}
layer {
  name: "relu_stage1_block0"
  type: "ReLU"
  bottom: "eltwise_stage1_block0"
  top: "eltwise_stage1_block0"
}
layer {
  name: "conv_stage1_block1_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage1_block0"
  top: "conv_stage1_block1_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block1_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage1_block1_branch2a"
  top: "conv_stage1_block1_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block1_branch2a"
  type: "Scale"
  bottom: "conv_stage1_block1_branch2a"
  top: "conv_stage1_block1_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block1_branch2a"
  type: "ReLU"
  bottom: "conv_stage1_block1_branch2a"
  top: "conv_stage1_block1_branch2a"
}
layer {
  name: "conv_stage1_block1_branch2b"
  type: "Convolution"
  bottom: "conv_stage1_block1_branch2a"
  top: "conv_stage1_block1_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block1_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage1_block1_branch2b"
  top: "conv_stage1_block1_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block1_branch2b"
  type: "Scale"
  bottom: "conv_stage1_block1_branch2b"
  top: "conv_stage1_block1_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block1_branch2b"
  type: "ReLU"
  bottom: "conv_stage1_block1_branch2b"
  top: "conv_stage1_block1_branch2b"
}
layer {
  name: "conv_stage1_block1_branch2c"
  type: "Convolution"
  bottom: "conv_stage1_block1_branch2b"
  top: "conv_stage1_block1_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block1_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage1_block1_branch2c"
  top: "conv_stage1_block1_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block1_branch2c"
  type: "Scale"
  bottom: "conv_stage1_block1_branch2c"
  top: "conv_stage1_block1_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage1_block1"
  type: "Eltwise"
  bottom: "eltwise_stage1_block0"
  bottom: "conv_stage1_block1_branch2c"
  top: "eltwise_stage1_block1"
}
layer {
  name: "relu_stage1_block1"
  type: "ReLU"
  bottom: "eltwise_stage1_block1"
  top: "eltwise_stage1_block1"
}
layer {
  name: "conv_stage1_block2_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage1_block1"
  top: "conv_stage1_block2_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block2_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage1_block2_branch2a"
  top: "conv_stage1_block2_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block2_branch2a"
  type: "Scale"
  bottom: "conv_stage1_block2_branch2a"
  top: "conv_stage1_block2_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block2_branch2a"
  type: "ReLU"
  bottom: "conv_stage1_block2_branch2a"
  top: "conv_stage1_block2_branch2a"
}
layer {
  name: "conv_stage1_block2_branch2b"
  type: "Convolution"
  bottom: "conv_stage1_block2_branch2a"
  top: "conv_stage1_block2_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block2_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage1_block2_branch2b"
  top: "conv_stage1_block2_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block2_branch2b"
  type: "Scale"
  bottom: "conv_stage1_block2_branch2b"
  top: "conv_stage1_block2_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block2_branch2b"
  type: "ReLU"
  bottom: "conv_stage1_block2_branch2b"
  top: "conv_stage1_block2_branch2b"
}
layer {
  name: "conv_stage1_block2_branch2c"
  type: "Convolution"
  bottom: "conv_stage1_block2_branch2b"
  top: "conv_stage1_block2_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block2_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage1_block2_branch2c"
  top: "conv_stage1_block2_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block2_branch2c"
  type: "Scale"
  bottom: "conv_stage1_block2_branch2c"
  top: "conv_stage1_block2_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage1_block2"
  type: "Eltwise"
  bottom: "eltwise_stage1_block1"
  bottom: "conv_stage1_block2_branch2c"
  top: "eltwise_stage1_block2"
}
layer {
  name: "relu_stage1_block2"
  type: "ReLU"
  bottom: "eltwise_stage1_block2"
  top: "eltwise_stage1_block2"
}
layer {
  name: "conv_stage1_block3_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage1_block2"
  top: "conv_stage1_block3_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block3_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage1_block3_branch2a"
  top: "conv_stage1_block3_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block3_branch2a"
  type: "Scale"
  bottom: "conv_stage1_block3_branch2a"
  top: "conv_stage1_block3_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block3_branch2a"
  type: "ReLU"
  bottom: "conv_stage1_block3_branch2a"
  top: "conv_stage1_block3_branch2a"
}
layer {
  name: "conv_stage1_block3_branch2b"
  type: "Convolution"
  bottom: "conv_stage1_block3_branch2a"
  top: "conv_stage1_block3_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block3_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage1_block3_branch2b"
  top: "conv_stage1_block3_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block3_branch2b"
  type: "Scale"
  bottom: "conv_stage1_block3_branch2b"
  top: "conv_stage1_block3_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage1_block3_branch2b"
  type: "ReLU"
  bottom: "conv_stage1_block3_branch2b"
  top: "conv_stage1_block3_branch2b"
}
layer {
  name: "conv_stage1_block3_branch2c"
  type: "Convolution"
  bottom: "conv_stage1_block3_branch2b"
  top: "conv_stage1_block3_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage1_block3_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage1_block3_branch2c"
  top: "conv_stage1_block3_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage1_block3_branch2c"
  type: "Scale"
  bottom: "conv_stage1_block3_branch2c"
  top: "conv_stage1_block3_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage1_block3"
  type: "Eltwise"
  bottom: "eltwise_stage1_block2"
  bottom: "conv_stage1_block3_branch2c"
  top: "eltwise_stage1_block3"
}
layer {
  name: "relu_stage1_block3"
  type: "ReLU"
  bottom: "eltwise_stage1_block3"
  top: "eltwise_stage1_block3"
}
layer {
  name: "conv_stage2_block0_proj_shortcut"
  type: "Convolution"
  bottom: "eltwise_stage1_block3"
  top: "conv_stage2_block0_proj_shortcut"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block0_proj_shortcut"
  type: "BatchNorm"
  bottom: "conv_stage2_block0_proj_shortcut"
  top: "conv_stage2_block0_proj_shortcut"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block0_proj_shortcut"
  type: "Scale"
  bottom: "conv_stage2_block0_proj_shortcut"
  top: "conv_stage2_block0_proj_shortcut"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv_stage2_block0_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage1_block3"
  top: "conv_stage2_block0_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block0_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block0_branch2a"
  top: "conv_stage2_block0_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block0_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block0_branch2a"
  top: "conv_stage2_block0_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block0_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block0_branch2a"
  top: "conv_stage2_block0_branch2a"
}
layer {
  name: "conv_stage2_block0_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block0_branch2a"
  top: "conv_stage2_block0_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block0_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block0_branch2b"
  top: "conv_stage2_block0_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block0_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block0_branch2b"
  top: "conv_stage2_block0_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block0_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block0_branch2b"
  top: "conv_stage2_block0_branch2b"
}
layer {
  name: "conv_stage2_block0_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block0_branch2b"
  top: "conv_stage2_block0_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block0_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block0_branch2c"
  top: "conv_stage2_block0_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block0_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block0_branch2c"
  top: "conv_stage2_block0_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block0"
  type: "Eltwise"
  bottom: "conv_stage2_block0_proj_shortcut"
  bottom: "conv_stage2_block0_branch2c"
  top: "eltwise_stage2_block0"
}
layer {
  name: "relu_stage2_block0"
  type: "ReLU"
  bottom: "eltwise_stage2_block0"
  top: "eltwise_stage2_block0"
}
layer {
  name: "conv_stage2_block1_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block0"
  top: "conv_stage2_block1_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block1_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block1_branch2a"
  top: "conv_stage2_block1_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block1_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block1_branch2a"
  top: "conv_stage2_block1_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block1_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block1_branch2a"
  top: "conv_stage2_block1_branch2a"
}
layer {
  name: "conv_stage2_block1_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block1_branch2a"
  top: "conv_stage2_block1_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block1_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block1_branch2b"
  top: "conv_stage2_block1_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block1_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block1_branch2b"
  top: "conv_stage2_block1_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block1_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block1_branch2b"
  top: "conv_stage2_block1_branch2b"
}
layer {
  name: "conv_stage2_block1_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block1_branch2b"
  top: "conv_stage2_block1_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block1_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block1_branch2c"
  top: "conv_stage2_block1_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block1_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block1_branch2c"
  top: "conv_stage2_block1_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block1"
  type: "Eltwise"
  bottom: "eltwise_stage2_block0"
  bottom: "conv_stage2_block1_branch2c"
  top: "eltwise_stage2_block1"
}
layer {
  name: "relu_stage2_block1"
  type: "ReLU"
  bottom: "eltwise_stage2_block1"
  top: "eltwise_stage2_block1"
}
layer {
  name: "conv_stage2_block2_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block1"
  top: "conv_stage2_block2_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block2_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block2_branch2a"
  top: "conv_stage2_block2_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block2_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block2_branch2a"
  top: "conv_stage2_block2_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block2_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block2_branch2a"
  top: "conv_stage2_block2_branch2a"
}
layer {
  name: "conv_stage2_block2_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block2_branch2a"
  top: "conv_stage2_block2_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block2_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block2_branch2b"
  top: "conv_stage2_block2_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block2_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block2_branch2b"
  top: "conv_stage2_block2_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block2_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block2_branch2b"
  top: "conv_stage2_block2_branch2b"
}
layer {
  name: "conv_stage2_block2_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block2_branch2b"
  top: "conv_stage2_block2_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block2_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block2_branch2c"
  top: "conv_stage2_block2_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block2_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block2_branch2c"
  top: "conv_stage2_block2_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block2"
  type: "Eltwise"
  bottom: "eltwise_stage2_block1"
  bottom: "conv_stage2_block2_branch2c"
  top: "eltwise_stage2_block2"
}
layer {
  name: "relu_stage2_block2"
  type: "ReLU"
  bottom: "eltwise_stage2_block2"
  top: "eltwise_stage2_block2"
}
layer {
  name: "conv_stage2_block3_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block2"
  top: "conv_stage2_block3_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block3_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block3_branch2a"
  top: "conv_stage2_block3_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block3_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block3_branch2a"
  top: "conv_stage2_block3_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block3_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block3_branch2a"
  top: "conv_stage2_block3_branch2a"
}
layer {
  name: "conv_stage2_block3_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block3_branch2a"
  top: "conv_stage2_block3_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block3_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block3_branch2b"
  top: "conv_stage2_block3_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block3_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block3_branch2b"
  top: "conv_stage2_block3_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block3_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block3_branch2b"
  top: "conv_stage2_block3_branch2b"
}
layer {
  name: "conv_stage2_block3_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block3_branch2b"
  top: "conv_stage2_block3_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block3_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block3_branch2c"
  top: "conv_stage2_block3_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block3_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block3_branch2c"
  top: "conv_stage2_block3_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block3"
  type: "Eltwise"
  bottom: "eltwise_stage2_block2"
  bottom: "conv_stage2_block3_branch2c"
  top: "eltwise_stage2_block3"
}
layer {
  name: "relu_stage2_block3"
  type: "ReLU"
  bottom: "eltwise_stage2_block3"
  top: "eltwise_stage2_block3"
}
layer {
  name: "conv_stage2_block4_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block3"
  top: "conv_stage2_block4_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block4_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block4_branch2a"
  top: "conv_stage2_block4_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block4_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block4_branch2a"
  top: "conv_stage2_block4_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block4_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block4_branch2a"
  top: "conv_stage2_block4_branch2a"
}
layer {
  name: "conv_stage2_block4_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block4_branch2a"
  top: "conv_stage2_block4_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block4_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block4_branch2b"
  top: "conv_stage2_block4_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block4_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block4_branch2b"
  top: "conv_stage2_block4_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block4_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block4_branch2b"
  top: "conv_stage2_block4_branch2b"
}
layer {
  name: "conv_stage2_block4_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block4_branch2b"
  top: "conv_stage2_block4_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block4_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block4_branch2c"
  top: "conv_stage2_block4_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block4_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block4_branch2c"
  top: "conv_stage2_block4_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block4"
  type: "Eltwise"
  bottom: "eltwise_stage2_block3"
  bottom: "conv_stage2_block4_branch2c"
  top: "eltwise_stage2_block4"
}
layer {
  name: "relu_stage2_block4"
  type: "ReLU"
  bottom: "eltwise_stage2_block4"
  top: "eltwise_stage2_block4"
}
layer {
  name: "conv_stage2_block5_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block4"
  top: "conv_stage2_block5_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block5_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage2_block5_branch2a"
  top: "conv_stage2_block5_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block5_branch2a"
  type: "Scale"
  bottom: "conv_stage2_block5_branch2a"
  top: "conv_stage2_block5_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block5_branch2a"
  type: "ReLU"
  bottom: "conv_stage2_block5_branch2a"
  top: "conv_stage2_block5_branch2a"
}
layer {
  name: "conv_stage2_block5_branch2b"
  type: "Convolution"
  bottom: "conv_stage2_block5_branch2a"
  top: "conv_stage2_block5_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block5_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage2_block5_branch2b"
  top: "conv_stage2_block5_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block5_branch2b"
  type: "Scale"
  bottom: "conv_stage2_block5_branch2b"
  top: "conv_stage2_block5_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage2_block5_branch2b"
  type: "ReLU"
  bottom: "conv_stage2_block5_branch2b"
  top: "conv_stage2_block5_branch2b"
}
layer {
  name: "conv_stage2_block5_branch2c"
  type: "Convolution"
  bottom: "conv_stage2_block5_branch2b"
  top: "conv_stage2_block5_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage2_block5_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage2_block5_branch2c"
  top: "conv_stage2_block5_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage2_block5_branch2c"
  type: "Scale"
  bottom: "conv_stage2_block5_branch2c"
  top: "conv_stage2_block5_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage2_block5"
  type: "Eltwise"
  bottom: "eltwise_stage2_block4"
  bottom: "conv_stage2_block5_branch2c"
  top: "eltwise_stage2_block5"
}
layer {
  name: "relu_stage2_block5"
  type: "ReLU"
  bottom: "eltwise_stage2_block5"
  top: "eltwise_stage2_block5"
}
layer {
  name: "conv_stage3_block0_proj_shortcut"
  type: "Convolution"
  bottom: "eltwise_stage2_block5"
  top: "conv_stage3_block0_proj_shortcut"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block0_proj_shortcut"
  type: "BatchNorm"
  bottom: "conv_stage3_block0_proj_shortcut"
  top: "conv_stage3_block0_proj_shortcut"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block0_proj_shortcut"
  type: "Scale"
  bottom: "conv_stage3_block0_proj_shortcut"
  top: "conv_stage3_block0_proj_shortcut"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv_stage3_block0_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage2_block5"
  top: "conv_stage3_block0_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block0_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage3_block0_branch2a"
  top: "conv_stage3_block0_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block0_branch2a"
  type: "Scale"
  bottom: "conv_stage3_block0_branch2a"
  top: "conv_stage3_block0_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block0_branch2a"
  type: "ReLU"
  bottom: "conv_stage3_block0_branch2a"
  top: "conv_stage3_block0_branch2a"
}
layer {
  name: "conv_stage3_block0_branch2b"
  type: "Convolution"
  bottom: "conv_stage3_block0_branch2a"
  top: "conv_stage3_block0_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block0_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage3_block0_branch2b"
  top: "conv_stage3_block0_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block0_branch2b"
  type: "Scale"
  bottom: "conv_stage3_block0_branch2b"
  top: "conv_stage3_block0_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block0_branch2b"
  type: "ReLU"
  bottom: "conv_stage3_block0_branch2b"
  top: "conv_stage3_block0_branch2b"
}
layer {
  name: "conv_stage3_block0_branch2c"
  type: "Convolution"
  bottom: "conv_stage3_block0_branch2b"
  top: "conv_stage3_block0_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block0_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage3_block0_branch2c"
  top: "conv_stage3_block0_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block0_branch2c"
  type: "Scale"
  bottom: "conv_stage3_block0_branch2c"
  top: "conv_stage3_block0_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage3_block0"
  type: "Eltwise"
  bottom: "conv_stage3_block0_proj_shortcut"
  bottom: "conv_stage3_block0_branch2c"
  top: "eltwise_stage3_block0"
}
layer {
  name: "relu_stage3_block0"
  type: "ReLU"
  bottom: "eltwise_stage3_block0"
  top: "eltwise_stage3_block0"
}
layer {
  name: "conv_stage3_block1_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage3_block0"
  top: "conv_stage3_block1_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block1_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage3_block1_branch2a"
  top: "conv_stage3_block1_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block1_branch2a"
  type: "Scale"
  bottom: "conv_stage3_block1_branch2a"
  top: "conv_stage3_block1_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block1_branch2a"
  type: "ReLU"
  bottom: "conv_stage3_block1_branch2a"
  top: "conv_stage3_block1_branch2a"
}
layer {
  name: "conv_stage3_block1_branch2b"
  type: "Convolution"
  bottom: "conv_stage3_block1_branch2a"
  top: "conv_stage3_block1_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block1_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage3_block1_branch2b"
  top: "conv_stage3_block1_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block1_branch2b"
  type: "Scale"
  bottom: "conv_stage3_block1_branch2b"
  top: "conv_stage3_block1_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block1_branch2b"
  type: "ReLU"
  bottom: "conv_stage3_block1_branch2b"
  top: "conv_stage3_block1_branch2b"
}
layer {
  name: "conv_stage3_block1_branch2c"
  type: "Convolution"
  bottom: "conv_stage3_block1_branch2b"
  top: "conv_stage3_block1_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block1_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage3_block1_branch2c"
  top: "conv_stage3_block1_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block1_branch2c"
  type: "Scale"
  bottom: "conv_stage3_block1_branch2c"
  top: "conv_stage3_block1_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage3_block1"
  type: "Eltwise"
  bottom: "eltwise_stage3_block0"
  bottom: "conv_stage3_block1_branch2c"
  top: "eltwise_stage3_block1"
}
layer {
  name: "relu_stage3_block1"
  type: "ReLU"
  bottom: "eltwise_stage3_block1"
  top: "eltwise_stage3_block1"
}
layer {
  name: "conv_stage3_block2_branch2a"
  type: "Convolution"
  bottom: "eltwise_stage3_block1"
  top: "conv_stage3_block2_branch2a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block2_branch2a"
  type: "BatchNorm"
  bottom: "conv_stage3_block2_branch2a"
  top: "conv_stage3_block2_branch2a"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block2_branch2a"
  type: "Scale"
  bottom: "conv_stage3_block2_branch2a"
  top: "conv_stage3_block2_branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block2_branch2a"
  type: "ReLU"
  bottom: "conv_stage3_block2_branch2a"
  top: "conv_stage3_block2_branch2a"
}
layer {
  name: "conv_stage3_block2_branch2b"
  type: "Convolution"
  bottom: "conv_stage3_block2_branch2a"
  top: "conv_stage3_block2_branch2b"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block2_branch2b"
  type: "BatchNorm"
  bottom: "conv_stage3_block2_branch2b"
  top: "conv_stage3_block2_branch2b"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block2_branch2b"
  type: "Scale"
  bottom: "conv_stage3_block2_branch2b"
  top: "conv_stage3_block2_branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu_stage3_block2_branch2b"
  type: "ReLU"
  bottom: "conv_stage3_block2_branch2b"
  top: "conv_stage3_block2_branch2b"
}
layer {
  name: "conv_stage3_block2_branch2c"
  type: "Convolution"
  bottom: "conv_stage3_block2_branch2b"
  top: "conv_stage3_block2_branch2c"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bn_stage3_block2_branch2c"
  type: "BatchNorm"
  bottom: "conv_stage3_block2_branch2c"
  top: "conv_stage3_block2_branch2c"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale_stage3_block2_branch2c"
  type: "Scale"
  bottom: "conv_stage3_block2_branch2c"
  top: "conv_stage3_block2_branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "eltwise_stage3_block2"
  type: "Eltwise"
  bottom: "eltwise_stage3_block1"
  bottom: "conv_stage3_block2_branch2c"
  top: "eltwise_stage3_block2"
}
layer {
  name: "relu_stage3_block2"
  type: "ReLU"
  bottom: "eltwise_stage3_block2"
  top: "eltwise_stage3_block2"
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "eltwise_stage3_block2"
  top: "pool"
  pooling_param {
    pool: AVE
    kernel_size: 7
    stride: 1
  }
}
layer {
   name: "fc_nsfw"
   type: "InnerProduct"
   bottom: "pool"
   top: "fc_nsfw"
   param {
       lr_mult: 5
       decay_mult: 1
   }
   param {
       lr_mult: 10
       decay_mult: 0
   }
   inner_product_param{
	   num_output: 2
	   weight_filler {
		 type: "xavier"
		 std: 0.01
	   }
	   bias_filler {
	      type: "xavier"
		  value: 0
	   }
   }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc_nsfw"
  top: "prob"
}

