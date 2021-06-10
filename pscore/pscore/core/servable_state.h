#pragma once

namespace pscore {

struct ServableState {
  enum class LiveState : int {
    kStart,
    kLoading,
    kAvailable,
    kUnloading,
    kEnd,
  };
};

}  // namespace pscore
