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

#include "lite/backends/metal/mps_conv_datasource.h"

@implementation MPSConvDataSource

- (MPSDataType)dataType API_AVAILABLE(ios(11.0)) {
    return MPSDataTypeFloat16;
}

- (MPSCNNConvolutionDescriptor* __nonnull)descriptor API_AVAILABLE(ios(10.0)) {
    return _descriptor;
}

- (void*)weights {
    return _weights;
}

- (float*)biasTerms {
    return _biasTerms;
}

- (BOOL)load {
    return YES;
}

- (void)purge {
}

- (NSString*)label {
    return @"mps_conv_add_relu_label";
}

- (nonnull id)copyWithZone:(nullable NSZone*)zone {
    return self;
}

@end
