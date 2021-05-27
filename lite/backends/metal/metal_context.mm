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

#include "lite/backends/metal/metal_context.h"
#include "lite/backends/metal/metal_context_imp.h"

namespace paddle {
namespace lite {

MetalContext::MetalContext() {
    mContext = (__bridge_retained void*)[[MetalContextImp alloc] init];
    if (mContext) {
        got_devices_ = true;
    }
}

MetalContext::~MetalContext() {
    CFRelease(mContext);
    mContext = nullptr;
}

void MetalContext::PrepareDevices() {
    if (got_devices_) return;
}

int MetalContext::GetDevicesNum() {
    if (!got_devices_) {
        return 0;
    }
    return 1;
}

void* MetalContext::GetDeviceByID(int id) {
    return nullptr;
}

void MetalContext::CreateCommandBuffer(RuntimeProgram* program) {
    program_ = program;
}

void MetalContext::WaitAllCompleted() {
    [(__bridge MetalContextImp*)mContext waitAllCompleted];
}

const void* MetalContext::GetDefaultDevice() {
    return nullptr;
}

void MetalContext::set_metal_path(std::string path) {
    metal_path_ = path;
    [(__bridge MetalContextImp*)mContext setMetalPath:path];
}
}
}
