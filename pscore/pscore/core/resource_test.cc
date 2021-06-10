#include "pscore/core/resource.h"
#include <gtest/gtest.h>

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

TEST(Resource, basic) {
  ResourceTest resource("ResourceTest", 100, 200);
  ResourceTest consume("ResourceTest", 10, 20);

  ASSERT_TRUE(resource.TryConsume(consume));
}

}  // namespace pscore