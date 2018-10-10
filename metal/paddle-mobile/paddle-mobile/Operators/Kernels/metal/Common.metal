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

#include <metal_stdlib>
using namespace metal;


inline void xyzn2abcd_1(int xyzn[4], int abcd[4]) {
  abcd[0] = abcd[1] = abcd[2] = 0;
  abcd[3] = xyzn[0] * 4 + xyzn[3];
}
inline void xyzn2abcd_2(int xyzn[4], int abcd[4]) {
  abcd[0] = abcd[1] = 0;
  abcd[2] = xyzn[1];
  abcd[3] = xyzn[0] * 4 + xyzn[3];
}
inline void xyzn2abcd_3(int xyzn[4], int abcd[4]) {
  abcd[0] = 0;
  abcd[3] = xyzn[0];
  abcd[2] = xyzn[1];
  abcd[1] = xyzn[2] * 4 + xyzn[3];
}
inline void xyzn2abcd_4(int C, int xyzn[4], int abcd[4]) {
  abcd[2] = xyzn[0];
  abcd[1] = xyzn[1];
  uint t = xyzn[2] * 4 + xyzn[3];
  abcd[0] = t / C;
  abcd[3] = t % C;
}

inline void abcd2xyzn_1(int abcd[4], int xyzn[4]) {
  xyzn[1] = xyzn[2] = 0;
  xyzn[0] = abcd[3] / 4;
  xyzn[1] = abcd[3] % 4;
}
inline void abcd2xyzn_2(int abcd[4], int xyzn[4]) {
  xyzn[2] = 0;
  xyzn[1] = abcd[2];
  xyzn[0] = abcd[3] / 4;
  xyzn[3] = abcd[3] % 4;
}
inline void abcd2xyzn_3(int abcd[4], int xyzn[4]) {
  xyzn[0] = abcd[3];
  xyzn[1] = abcd[2];
  xyzn[2] = abcd[1] / 4;
  xyzn[3] = abcd[1] % 4;
}
inline void abcd2xyzn_4(int C, int abcd[4], int xyzn[4]) {
  xyzn[0] = abcd[2];
  xyzn[1] = abcd[1];
  uint t = abcd[0] * C + abcd[3];
  xyzn[2] = t / 4;
  xyzn[3] = t % 4;
}

inline void xyzn2abcd(int C, int xyzn[4], int abcd[4]) {
  abcd[2] = xyzn[0];
  abcd[1] = xyzn[1];
  uint t = xyzn[2] * 4 + xyzn[3];
  abcd[0] = t / C;
  abcd[3] = t % C;
}

inline void abcd2xyzn(int C, int abcd[4], int xyzn[4]) {
  xyzn[0] = abcd[2];
  xyzn[1] = abcd[1];
  uint t = abcd[0] * C + abcd[3];
  xyzn[2] = t / 4;
  xyzn[3] = t % 4;
}

inline int32_t abcd2index(int32_t dim[4], int32_t abcd[4]) {
  int32_t r = abcd[0];
  r = r * dim[1] + abcd[1];
  r = r * dim[2] + abcd[2];
  r = r * dim[3] + abcd[3];
  return r;
}

inline void index2abcd(int32_t dim[4], int32_t ind, int32_t abcd[4]) {
  abcd[3] = ind % dim[3]; ind /= dim[3];
  abcd[2] = ind % dim[2]; ind /= dim[2];
  abcd[1] = ind % dim[1]; ind /= dim[1];
  abcd[0] = ind;
}

inline void trans(int32_t trans[4], int32_t ipos[4], int32_t opos[4]) {
  for (int i = 0; i < 4; i++) {
    opos[i] = ipos[trans[i]];
  }
}

inline void invtrans(int32_t trans[4], int32_t ipos[4], int32_t opos[4]) {
  for (int i = 0; i < 4; i++) {
    opos[trans[i]] = ipos[i];
  }
}


struct MetalConvParam {
  short offsetX;
  short offsetY;
  short offsetZ;
  ushort strideX;
  ushort strideY;
  ushort dilationX;
  ushort dilationY;
};

