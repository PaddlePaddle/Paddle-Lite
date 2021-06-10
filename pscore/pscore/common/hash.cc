#include "pscore/common/hash.h"

namespace pscore {

uint64_t HashCombine(const uint64_t hash0, const uint64_t hash1) {
  return hash0 ^ (hash1 + 0x9e3779b97f4a7800 + (hash0 << 10) + (hash0 >> 4));
}

}  // namespace pscore
