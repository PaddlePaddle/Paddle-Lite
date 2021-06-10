#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/strings/string_view.h>
#include <absl/types/span.h>
#include "pscore/core/resource.h"

namespace pscore {

/**
 * A manager helps to record multiple kinds of resource.
 */
class ResourceManager {
 public:
  //! Register a kind of resource named \p name and size is \p size. This
  //! happens in the initialization of the manager.
  void Register(absl::string_view name, float size);

  //! Register some Resources which bind to the same resource kind.
  /// This is used when some resource like GPU is used.
  void Register(absl::string_view kind, absl::Span<Resource> resources);

  //! Consume some resource. This happens when a Servable is going to load.
  bool TryConsume(absl::string_view name, float size);

  bool TryConsume(absl::string_view name, uint32_t id, float size);

  //! Try to consume a combination of different kinds of resources.
  /// Such as {300M host memory, 1000M GPU memory, 1 CPU core} and so on.
  bool TryConsume(absl::Span<Resource> resources);

  //! Reclaim some resource. This happens when a Servable is unloaded and return
  //! all the resources.
  bool Reclaim(absl::string_view name, float size);

 private:
  absl::flat_hash_map<std::string,
                      absl::InlinedVector<std::unique_ptr<Resource>, 2>>
      data_;
};

}  // namespace pscore
