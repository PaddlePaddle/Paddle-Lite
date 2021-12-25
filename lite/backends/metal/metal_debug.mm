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

#include <cstdio>
#include <cstdlib>

#include "lite/backends/metal/metal_debug.h"
#include "lite/backends/metal/target_wrapper.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

void MetalDebug::print_log(const std::string& name, const MetalImage* metalImg, int inCount) {
    auto size = metalImg->tensor_dim_.production();
    float* data = (float*)TargetWrapperMetal::Malloc(sizeof(float) * size);
    metalImg->template CopyToNCHW<float>(data);
    std::string log_string = name + " ";
    for (int i = 0; i < metalImg->tensor_dim_.size(); i++) {
        if (i == 0) {
            log_string = log_string + "[" + std::to_string(metalImg->tensor_dim_[i]);
            if (metalImg->tensor_dim_.size() != 1) {
                log_string += ", ";
            } else {
                log_string += "]";
            }
        } else if (i == metalImg->tensor_dim_.size() - 1) {
            log_string = log_string + std::to_string(metalImg->tensor_dim_[i]) + "]";
        } else {
            log_string = log_string + std::to_string(metalImg->tensor_dim_[i]) + ", ";
        }
    }
    print_float(log_string, data, (int)size, inCount);
    TargetWrapperMetal::Free(data);
}

void MetalDebug::print_log(const std::string& name, MetalBuffer* metalBuf, int inCount) {
    auto size = metalBuf->tensor_dim().production();
    float* data = (float*)TargetWrapperMetal::Malloc(metalBuf->mtl_size());
    metalBuf->template CopyToNCHW<float>(data);
    print_float(name, data, (int)size, inCount);
    TargetWrapperMetal::Free(data);
}

void MetalDebug::print_float(const std::string& name, float* data, int size, int inCount) {
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(data[i], i);
    }
    if (vec.size() < inCount) {
        inCount = (int)vec.size();
    }
    int stride = (int)size / inCount;
    int realCount = (int)size / stride;
    layer_count_++;
    NSLog(@"---------------------------------------------------");
    NSLog(@"%d : %@",
        layer_count_,
        [NSString stringWithCString:name.c_str() encoding:NSASCIIStringEncoding]);
    for (int i = 0; i < realCount; i++) {
        float value = vec[i * stride].first;
        int index = vec[i * stride].second;
        if (i == 0) {
            printf("[(%d: %lf),", index, value);
        } else if (i == realCount - 1) {
            printf(" (%d: %lf)]\n", index, value);
        } else {
            printf(" (%d: %lf),", index, value);
        }
    }
    if (name == "fetch") {
        layer_count_ = 0;
        NSLog(@"====================================================");
    }
}

void MetalDebug::SaveOutput_(std::string name, MetalImage* image, DumpMode mode) {
    if (name == "feed" || name == "fetch") return;
    layer_count_++;
    if (op_stats_.count(name) > 0) {
        op_stats_[name] += 1;
        auto name_plus_index =
            std::to_string(layer_count_) + "-" + name + "-" + std::to_string(op_stats_[name]);
        DumpImage(name_plus_index, image, mode);
    } else {
        op_stats_[name] = 1;
        auto name_plus_index =
            std::to_string(layer_count_) + "-" + name + "-" + std::to_string(op_stats_[name]);
        DumpImage(name_plus_index, image, mode);
    }
}

void MetalDebug::DumpImage(const std::string& name, MetalImage* image, DumpMode mode) {
    auto length = image->tensor_dim_.production();
    auto buf = (float*)TargetWrapperMetal::Malloc(sizeof(float) * length);
    image->CopyToNCHW<float>(buf);

    std::string filename = name + ".txt";
    FILE* fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < length; ++i) {
        if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
        if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
    }
    TargetWrapperMetal::Free(buf);
    fclose(fp);
}

void MetalDebug::DumpBuffer(const std::string& name, MetalBuffer* buffer, DumpMode mode) {
    int64_t length = buffer->tensor_dim().production();
    auto buf = (float*)TargetWrapperMetal::Malloc(sizeof(float) * length);
    std::string filename = name + ".txt";
    FILE* fp = fopen(filename.c_str(), "w");
    buffer->CopyToNCHW<float>(buf);
    for (int i = 0; i < length; ++i) {
        if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
        if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
    }
    TargetWrapperMetal::Free(buf);
    fclose(fp);
}

LITE_THREAD_LOCAL std::map<std::string, int> MetalDebug::op_stats_ = {};
LITE_THREAD_LOCAL bool MetalDebug::enable_ = false;
LITE_THREAD_LOCAL int MetalDebug::layer_count_ = 0;

}  // namespace lite
}  // namespace paddle
