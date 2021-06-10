#pragma once

#include <absl/strings/str_format.h>
#include <absl/strings/string_view.h>
#include "pscore/common/macros.h"

namespace pscore {

/**
 * An standard abstraction for resources those should be sensed by the system.
 */
class Resource {
 public:
  Resource(absl::string_view name) : name_(name) {}

  //! Try to consume some resource, return true if succeed or false elsewise.
  /// One can cast the \p consumption to a specific Resource type.
  virtual bool TryConsume(const Resource& consumption) = 0;

  //! To return some Resource.
  virtual bool Recycle(const Resource& consumption) = 0;

  std::string ToString() const {
    return absl::StrFormat("{Resource %s size: %f}", name_, size_);
  }

 private:
  std::string name_;
  float size_;
};

}  // namespace pscore
