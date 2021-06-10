#pragma once

#include "pscore/common/macros.h"
#include "pscore/core/status.h"
#include "pscore/core/any_ptr.h"

namespace pscore {

/**
 * An abstraction for an object that manages the lifecycle of a Servable.
 */
class Loader {
 public:
  virtual ~Loader() = default;

  //! Fetches any resource that needs for a servable.
  virtual Status Load() {
    return errors::Unimplemented("Load isn't implemented");
  }

  //! Frees any resources allocated during the live time of the target Servable.
  virtual void Unload() = 0;

  //! Returns an opaque interface to the underlying servable object.
  virtual AnyPtr servable() = 0;

 private:
};

}  // namespace pscore
