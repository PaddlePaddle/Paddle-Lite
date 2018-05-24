#pragma once

#include <map>
#include <string>

#include "framework/operator.h"
#include "node.h"

namespace paddle_mobile {
namespace framework {

class FusionOpRegister {
 public:
  static FusionOpRegister *Instance() {
    static FusionOpRegister *regist = nullptr;
    if (regist == nullptr) {
      regist = new FusionOpRegister();
    }
    return regist;
  }

  void regist(FusionOpMatcher* matcher) {
    std::shared_ptr<FusionOpMatcher> shared_matcher(matcher);
    matchers_[matcher->Type()] = shared_matcher;
  }

  const std::map<std::string, std::shared_ptr<FusionOpMatcher>> Matchers() {
    return matchers_;
  }

 private:
  std::map<std::string, std::shared_ptr<FusionOpMatcher>> matchers_;
  FusionOpRegister() {}
};

class FusionOpRegistrar{
 public:
  explicit FusionOpRegistrar(FusionOpMatcher* matcher){
    FusionOpRegister::Instance()->regist(matcher);
  }
};

}  // namespace framework
}  // namespace paddle_mobile
