// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <sys/time.h>

#include <iostream>
#include <string>

#include "lite/utils/log/cp_logging.h"

#ifdef USE_METAL_STATS
#define METAL_TIME_STATS(object_name, stat_name) MetalTimeStats object_name(stat_name);
#else
#define METAL_TIME_STATS(object_name, stat_name)
#endif

class MetalTimeStats {
   public:
    explicit MetalTimeStats(std::string name) : name_(name) {
        gettimeofday(&start_, NULL);
    }

    virtual ~MetalTimeStats() {
        gettimeofday(&end_, NULL);
        double run_time = (end_.tv_sec - start_.tv_sec) * 1.0e6 +
                          static_cast<double>(end_.tv_usec - start_.tv_usec);
        LOG(INFO) << "Module " << name_ << " run_time is " << run_time << " us "
                  << "\n";
    }

   private:
    struct timeval start_;
    struct timeval end_;
    struct timeval system_start;
    std::string name_;
};
