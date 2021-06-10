#pragma once
#include <absl/strings/str_format.h>
#include <string>

namespace pscore {

struct ServableId {
  // The name of the servable.
  std::string name;

  // The version of the servable object.
  int64_t version;

  ServableId(absl::string_view name, int64_t version)
      : name(name), version(version) {}

  bool operator==(const ServableId& o) const {
    return name == o.name && version == o.version;
  }
  bool operator!=(const ServableId& o) const { return !(*this == o); }

  std::string DebugString() const {
    return absl::StrFormat("{name:%s version:%d}", name, version);
  }
};

}  // namespace pscore
