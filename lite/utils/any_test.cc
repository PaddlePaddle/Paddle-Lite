#include "lite/utils/any.h"
#include "lite/core/tensor.h"

int main() {
  using namespace paddle::lite;

  Any any_0;
  any_0.set<int>(12);
  CHECK_EQ(any_0.get<int>(), 12);

  Any any_1;
  Tensor tensor_1;
  any_0.set<Tensor>(std::move(tensor_1));
  CHECK(any_0.valid());

  return 0;
}