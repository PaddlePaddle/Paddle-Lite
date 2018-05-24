#pragma once

#include <string>
#include "stdio.h"

namespace paddle_mobile {

class PaddleMobileObject {
 public:
  virtual std::string ToString() {
    char address[128] = {0};
    sprintf(address, "%p", this);
    return std::string(address);
  }

 private:
};
}  // namespace paddle_mobile
