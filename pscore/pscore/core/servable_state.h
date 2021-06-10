#pragma once

#include "pscore/common/macros.h"
#include "pscore/core/servable_id.h"
#include "pscore/core/status.h"

namespace pscore {

struct ServableState {
  ServableId id;

  enum class State : int {
    kStart,
    kLoading,
    kAvailable,
    kUnloading,
    kEnd,
  };

  State state;

  // Whether anything has gone wrong with this servable.
  Status health;

  std::string DebugString() const;
};

std::string StateToStr(ServableState state);

}  // namespace pscore
