#include "pscore/core/servable_state.h"

namespace pscore {

std::string StateToStr(ServableState::State state) {
  switch (state) {
    case ServableState::State::kStart:
      return "Start";
    case ServableState::State::kLoading:
      return "Loading";
    case ServableState::State::kAvailable:
      return "Available";
    case ServableState::State::kUnloading:
      return "Unloading";
    case ServableState::State::kEnd:
      return "End";
    default:
      PSCORE_NOT_IMPLEMENTED
  }
  return "";
}

std::string ServableState::DebugString() const {
  return absl::StrFormat("id: %s state: %s health: %s",
                         id.DebugString(),
                         StateToStr(state),
                         health.ToString());
}

}  // namespace pscore
