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

#include "fpga/V2/driver/bitmap.h"

namespace fpga_bitmap {
void bitmap_set(uint64_t *map, unsigned int start, int len) {
  uint64_t *p = map + BIT_WORD(start);
  const unsigned int size = start + len;
  int bits_to_set = BITS_PER_LONG - (start % BITS_PER_LONG);
  uint64_t mask_to_set = BITMAP_FIRST_WORD_MASK(start);

  while (len - bits_to_set >= 0) {
    *p |= mask_to_set;
    len -= bits_to_set;
    bits_to_set = BITS_PER_LONG;
    mask_to_set = ~0UL;
    p++;
  }
  if (len) {
    mask_to_set &= BITMAP_LAST_WORD_MASK(size);
    *p |= mask_to_set;
  }
}

void bitmap_clear(uint64_t *map, unsigned int start, int len) {
  uint64_t *p = map + BIT_WORD(start);
  const unsigned int size = start + len;
  int bits_to_clear = BITS_PER_LONG - (start % BITS_PER_LONG);
  uint64_t mask_to_clear = BITMAP_FIRST_WORD_MASK(start);

  while (len - bits_to_clear >= 0) {
    *p &= ~mask_to_clear;
    len -= bits_to_clear;
    bits_to_clear = BITS_PER_LONG;
    mask_to_clear = ~0UL;
    p++;
  }
  if (len) {
    mask_to_clear &= BITMAP_LAST_WORD_MASK(size);
    *p &= ~mask_to_clear;
  }
}

static uint64_t ffs(uint64_t data) {
  uint64_t bit = 0;
  int i = 0;

  for (i = 0; i < sizeof(data); i++) {
    if (data & (1 << i)) {
      bit = i;
      break;
    }
  }

  return bit;
}

static uint64_t _find_next_bit(const uint64_t *addr, uint64_t nbits,
                               uint64_t start, uint64_t invert) {
  uint64_t tmp = 0;

  if (!nbits || start >= nbits) return nbits;

  tmp = addr[start / BITS_PER_LONG] ^ invert;

  /* Handle 1st word. */
  tmp &= BITMAP_FIRST_WORD_MASK(start);
  start = round_down(start, BITS_PER_LONG);

  while (!tmp) {
    start += BITS_PER_LONG;
    if (start >= nbits) return nbits;

    tmp = addr[start / BITS_PER_LONG] ^ invert;
  }

  return (start + ffs(tmp)) < nbits ? (start + ffs(tmp)) : nbits;
}

uint64_t find_next_zero_bit(const uint64_t *addr, uint64_t size,
                            uint64_t offset) {
  return _find_next_bit(addr, size, offset, ~0UL);
}

uint64_t find_next_bit(const uint64_t *addr, uint64_t size, uint64_t offset) {
  return _find_next_bit(addr, size, offset, 0UL);
}

uint64_t bitmap_find_next_zero_area_off(uint64_t *map, uint64_t size,
                                        uint64_t start, unsigned int nr,
                                        uint64_t align_mask,
                                        uint64_t align_offset) {
  uint64_t index = 0;
  uint64_t end = 0;
  uint64_t i = 0;

again:
  index = find_next_zero_bit(map, size, start);

  /* Align allocation */
  index = __ALIGN_MASK(index + align_offset, align_mask) - align_offset;

  end = index + nr;
  if (end > size) return end;
  i = find_next_bit(map, end, index);
  if (i < end) {
    start = i + 1;
    goto again;
  }

  return index;
}

uint64_t bitmap_find_next_zero_area(uint64_t *map, uint64_t size,
                                    uint64_t start, unsigned int nr,
                                    uint64_t align_mask) {
  return bitmap_find_next_zero_area_off(map, size, start, nr, align_mask, 0);
}
}  // namespace fpga_bitmap
