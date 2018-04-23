#pragma once

#include <string>
#include "framework/program.h"

namespace paddle_mobile {

Program Load(const std::string &dirname);

class Executor {};
}
