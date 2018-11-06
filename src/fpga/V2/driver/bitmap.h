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

#pragma once

#include <stdint.h>
#include <stdio.h>

#define BITS_PER_LONG 64
#define BIT_WORD(nr) ((nr) / BITS_PER_LONG)
#define BITMAP_FIRST_WORD_MASK(start) (~0UL << ((start) & (BITS_PER_LONG - 1)))
#define BITMAP_LAST_WORD_MASK(nbits) (~0UL >> (-(nbits) & (BITS_PER_LONG - 1)))

#define __ALIGN_KERNEL_MASK(x, mask) (((x) + (mask)) & ~(mask))
#define __ALIGN_MASK(x, mask) __ALIGN_KERNEL_MASK((x), (mask))

#define round_down(x, y) ((x) & ((y)-1))

namespace fpga_bitmap {
void bitmap_set(uint64_t *map, unsigned int start, int len);
void bitmap_clear(uint64_t *map, unsigned int start, int len);
uint64_t bitmap_find_next_zero_area(uint64_t *map, uint64_t size,
                                    uint64_t start, unsigned int nr,
                                    uint64_t align_mask);

}  // namespace fpga_bitmap
