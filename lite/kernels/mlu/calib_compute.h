#pragma once
#include "lite/core/kernel.h"
#include "lite/operators/calib_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

class CalibComputeFp32ToInt8
    : public KernelLite<TARGET(kMLU), PRECISION(kInt8)> {
 public:
  using param_t = operators::CalibParam;

  void Run() override;

  ~CalibComputeFp32ToInt8() override{};

 private:
};

class CalibComputeInt8ToFp32
    : public KernelLite<TARGET(kMLU), PRECISION(kInt8)> {
 public:
  using param_t = operators::CalibParam;

  void Run() override;

  ~CalibComputeInt8ToFp32() override{};

 private:
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
