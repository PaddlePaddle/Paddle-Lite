//
//  common.metal
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/8/26.
//  Copyright © 2018年 orange. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

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
