#pragma once

#include "common/types.h"
#include "framework/paddle_mobile_object.h"
#include "framework/program/program_desc.h"
#include "framework/scope.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype, Precision P = Precision::FP32>
class Program : PaddleMobileObject {
 public:
  std::shared_ptr<ProgramDesc> originProgram;
  std::shared_ptr<ProgramDesc> optimizeProgram;
  std::shared_ptr<Scope> scope;

 private:
};

}  // namespace framework
}  // namespace paddle_mobile
