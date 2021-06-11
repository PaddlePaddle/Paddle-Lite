#include "pscore/core/resource.h"

namespace pscore {

namespace {

struct ResourceTest : public Resource {
  float size0;
  float size1;

  explicit ResourceTest(absl::string_view name, float size0, float size1)
      : Resource(name), size0(size0), size1(size1) {}

  bool TryConsume(const Resource &consumption) override {
    auto *r = static_cast<const ResourceTest *>(&consumption);

    if (size0 < r->size0 || size1 < r->size1) return false;

    size0 -= r->size0;
    size1 -= r->size1;

    return true;
  }

  bool Recycle(const Resource &consumption) override {
    auto *r = static_cast<const ResourceTest *>(&consumption);
    size0 += r->size0;
    size1 += r->size1;
    return true;
  }
};

}  // namespace



}  // namespace pscore

#define CATCH_CONFIG_MAIN

int main() {
  return 0;
}