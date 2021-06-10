#pragma once
#include <cstddef>

namespace pscore {

template <typename Type>
static size_t FastTypeId() {
  // Use a static variable to get a unique per-type address.
  static int dummy;
  return reinterpret_cast<std::size_t>(&dummy);
}

}  // namespace pscore
