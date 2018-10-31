/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#ifdef P

#include "Macro.metal"


#pragma mark - convAdd
kernel void FUNC3_(conv_add_1x1, PRELU_TYPE, P)(texture2d_array<P, access::sample> inTexture [[texture(0)]],
                         texture2d_array<P, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device VECTOR(P, 4) *weights [[buffer(1)]],
                         const device VECTOR(P, 4) *biase [[buffer(2)]],
#ifdef PRELU_CHANNEL
                         const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_ELEMENT
                         const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_OTHER
                         const device P *alpha [[buffer(3)]],
#endif
                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 1;
  
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  VECTOR(P, 4) output = biase[gid.z];
  
  VECTOR(P, 4) input;
  for (uint i = 0; i < input_arr_size; ++i) {
    input = inTexture.sample(sample,float2(posInInput.x, posInInput.y), i);
    VECTOR(P, 4) weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
    output.x += dot(input, weight_x);
    
    VECTOR(P, 4) weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
    output.y += dot(input, weight_y);
    
    VECTOR(P, 4) weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
    output.z += dot(input, weight_z);
    
    VECTOR(P, 4) weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
    output.w += dot(input, weight_w);
  }
  
//  output = output + float4(biase[gid.z]);
  
#ifdef PRELU_CHANNEL
  VECTOR(P, 4) alpha_value = alpha[gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
  int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
  VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
  P alpha_value = alpha[0];
  output.x = output.x > 0 ? output.x : (alpha_value * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
  outTexture.write(VECTOR(P, 4)(output), gid.xy, gid.z);
}

kernel void FUNC3_(conv_add_3x3, PRELU_TYPE, P)(texture2d_array<P, access::sample> inTexture [[texture(0)]],
    texture2d_array<P, access::write> outTexture [[texture(1)]],
    constant MetalConvParam &param [[buffer(0)]],
    const device VECTOR(P, 4) *weights [[buffer(1)]],
    const device VECTOR(P, 4) *biase [[buffer(2)]],
#ifdef PRELU_CHANNEL
     const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_ELEMENT
     const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_OTHER
     const device P *alpha [[buffer(3)]],
#endif
     uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }

  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

  const uint kernelHXW = 9;

  uint input_arr_size = inTexture.get_array_size();

  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

  VECTOR(P, 4) output = biase[gid.z];

  ushort dilation_x = param.dilationX;
  ushort dilation_y = param.dilationY;

  VECTOR(P, 4) input[9];

  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), i);

    input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - dilation_y), i);

    input[2] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), i);

    input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y), i);

    input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);

    input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y), i);

    input[6] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y + dilation_y), i);

    input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + dilation_y), i);

    input[8] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y + dilation_y), i);

    for (int j = 0; j < 9; ++j) {
      VECTOR(P, 4) weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);

      VECTOR(P, 4) weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);

      VECTOR(P, 4) weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);

      VECTOR(P, 4) weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
//  output = output + float4(biase[gid.z]);
  
#ifdef PRELU_CHANNEL
  VECTOR(P, 4) alpha_value = alpha[gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
  int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
  VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
  P alpha_value = alpha[0];
  output.x = output.x > 0 ? output.x : (alpha_value * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
  outTexture.write(VECTOR(P, 4)(output), gid.xy, gid.z);
}

kernel void FUNC3_(conv_add_5x1, PRELU_TYPE, P)(texture2d_array<P, access::sample> inTexture [[texture(0)]],
                         texture2d_array<P, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device VECTOR(P, 4) *weights [[buffer(1)]],
                         const device VECTOR(P, 4) *biase [[buffer(2)]],
#ifdef PRELU_CHANNEL
                        const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_ELEMENT
                        const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_OTHER
                        const device P *alpha [[buffer(3)]],
#endif
                         uint3 gid [[thread_position_in_grid]]) {

  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }

  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

  const uint kernelHXW = 5;

  uint input_arr_size = inTexture.get_array_size();

  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

  VECTOR(P, 4) output = biase[gid.z];;

  ushort dilation_y = param.dilationY;
  VECTOR(P, 4) input[5];

  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 2 * dilation_y), i);

    input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);

    input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

    input[3] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), i);

    input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 2 * dilation_y), i);

    for (int j = 0; j < 5; ++j) {
      VECTOR(P, 4) weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);

      VECTOR(P, 4) weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);

      VECTOR(P, 4) weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);

      VECTOR(P, 4) weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
  
#ifdef PRELU_CHANNEL
  VECTOR(P, 4) alpha_value = alpha[gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
  int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
  VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
  P alpha_value = alpha[0];
  output.x = output.x > 0 ? output.x : (alpha_value * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
  outTexture.write(VECTOR(P, 4)(output), gid.xy, gid.z);
}


kernel void FUNC3_(conv_add_1x5, PRELU_TYPE, P)(texture2d_array<P, access::sample> inTexture [[texture(0)]],
                         texture2d_array<P, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device VECTOR(P, 4) *weights [[buffer(1)]],
                         const device VECTOR(P, 4) *biase [[buffer(2)]],
#ifdef PRELU_CHANNEL
                         const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_ELEMENT
                         const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_OTHER
                         const device P *alpha [[buffer(3)]],
#endif
                         uint3 gid [[thread_position_in_grid]]) {

  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }

  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

  const uint kernelHXW = 5;

  uint input_arr_size = inTexture.get_array_size();

  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

  VECTOR(P, 4) output = biase[gid.z];

  ushort dilation_x = param.dilationX;
  VECTOR(P, 4) input[5];

  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x - 2 * dilation_x, posInInput.y), i);

    input[1] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);

    input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

    input[3] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), i);

    input[4] = inTexture.sample(sample, float2(posInInput.x + 2 * dilation_x, posInInput.y), i);

    for (int j = 0; j < 5; ++j) {
      VECTOR(P, 4) weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);

      VECTOR(P, 4) weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);

      VECTOR(P, 4) weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);

      VECTOR(P, 4) weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
  
#ifdef PRELU_CHANNEL
  VECTOR(P, 4) alpha_value = alpha[gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
  int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
  VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
  P alpha_value = alpha[0];
  output.x = output.x > 0 ? output.x : (alpha_value * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
  outTexture.write(VECTOR(P, 4)(output), gid.xy, gid.z);
}

kernel void FUNC3_(depthwise_conv_add_3x3, PRELU_TYPE, P)(texture2d_array<P, access::sample> inTexture [[texture(0)]],
    texture2d_array<P, access::write> outTexture [[texture(1)]],
    constant MetalConvParam &param [[buffer(0)]],
    const device P *weights [[buffer(1)]],
    const device VECTOR(P, 4) *biase [[buffer(2)]],
#ifdef PRELU_CHANNEL
    const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_ELEMENT
    const device VECTOR(P, 4) *alpha [[buffer(3)]],
#endif
#ifdef PRELU_OTHER
    const device P *alpha [[buffer(3)]],
#endif
    uint3 gid [[thread_position_in_grid]]) {

  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  uint output_slice = gid.z;
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint weithTo = gid.z * kernelHXW * 4;
  VECTOR(P, 4) output = biase[gid.z];
  VECTOR(P, 4) inputs[9];
  inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
  inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
  inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
  inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
  inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
  inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
  inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
  inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
  inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
  for (int j = 0; j < 9; ++j) {
    VECTOR(P, 4) input = inputs[j];
    output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
    output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
    output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
    output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
  }
  
#ifdef PRELU_CHANNEL
  VECTOR(P, 4) alpha_value = alpha[gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
  int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
  VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
  output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
  P alpha_value = alpha[0];
  output.x = output.x > 0 ? output.x : (alpha_value * output.x);
  output.y = output.y > 0 ? output.y : (alpha_value * output.y);
  output.z = output.z > 0 ? output.z : (alpha_value * output.z);
  output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
  outTexture.write(VECTOR(P, 4)(output), gid.xy, gid.z);
}

#endif

