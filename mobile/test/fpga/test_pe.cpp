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

#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#include "fpga/V2/filter.h"

namespace fpga = paddle_mobile::fpga;

static const uint32_t N = 64;
static const uint32_t C = 3;
static const uint32_t H = 224;
static const uint32_t W = 224;
static const uint32_t G = 1;

fpga::DataType input_type = fpga::DATA_TYPE_FP32;
fpga::DataType output_type = fpga::DATA_TYPE_FP16;

void* ifm = nullptr;
void* ofm = nullptr;
void* filter = nullptr;
void* ifm_scale = nullptr;
void* ofm_scale = nullptr;
void* filter_scale = nullptr;

int ifm_size = 0, ofm_size = 0;

void format_data() {
  ifm_scale = fpga::fpga_malloc(8);
  ofm_scale = fpga::fpga_malloc(8);
  int ifm_channel = fpga::filter::calc_aligned_channel(C);
  int ofm_channel = fpga::filter::calc_aligned_channel(N);
  int num = fpga::filter::calc_aligned_num(N, C);
  DLOG << "ifm_channel = " << ifm_channel;
  DLOG << "ofm_channel = " << ofm_channel;
  DLOG << "aligned_num = " << num;
  ifm_size = ifm_channel * H * W;
  ofm_size = ofm_channel * H * W;
  ifm = fpga::fpga_malloc(ifm_size * sizeof(float));
  ofm = fpga::fpga_malloc(ofm_size * sizeof(int16_t));
  memset(ifm, 0, ifm_size * sizeof(float));
  memset(ofm, 0, ofm_size * sizeof(int16_t));

  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      for (int c = 0; c < C; c++) {
        int index = h * W * ifm_channel + w * ifm_channel + c;
        (reinterpret_cast<float*>(ifm))[index] = h + w + c * 0.1f;
        // DLOG << index << ":" << ((float *) ifm)[index];
      }
    }
  }
  fpga::fpga_flush(ifm, ifm_size * sizeof(float));
  fpga::fpga_flush(ofm, ofm_size * sizeof(int16_t));
}

void print_fp16(int16_t* ptr, int total_size, int num) {
  fpga::fpga_invalidate(ptr, total_size * sizeof(int16_t));
  int stride = total_size / num;
  for (int i = 0; i < total_size; i += stride) {
    DLOG << fpga::fp16_2_fp32(ptr[i]);
  }
}

void print_fp32(float* ptr, int total_size, int num) {
  fpga::fpga_invalidate(ptr, total_size * sizeof(float));
  int stride = total_size / num;
  for (int i = 0; i < total_size; i += stride) {
    DLOG << ptr[i];
  }
}

void test_bypass() {
  fpga::BypassArgs args;
  args.input_data_type = input_type;
  args.output_data_type = output_type;
  args.image.address = ifm;
  args.image.height = H;
  args.image.width = W;
  args.image.channels = C;
  args.image.scale_address = reinterpret_cast<float*>(ifm_scale);
  args.output.address = ofm;
  args.output.scale_address = reinterpret_cast<float*>(ofm_scale);
  fpga::PerformBypass(args);
}

int main() {
  paddle_mobile::fpga::open_device();
  format_data();
  DLOG << "format data done";
  print_fp32(reinterpret_cast<float*>(ifm), ifm_size, 200);
  DLOG << "print input done";
  test_bypass();
  DLOG << "test done";
  print_fp16(reinterpret_cast<int16_t*>(ofm), ifm_size, 200);
  std::cout << "Computation done" << std::endl;
  return 0;
}

#endif
