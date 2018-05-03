
/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
#pragma once

#include <type_traits>
#include "types.h"

namespace paddle_mobile{
    namespace framework{

        struct __host_target{};
        struct __device_target{};

        struct __arm_device{};
        struct __cpu_device{};

        struct __HtoD{};
        struct __HtoH{};
        struct __DtoD{};
        struct __DtoH{};

        template <typename DeviceType>
        struct DeviceTraits {
            typedef __invalid_type target_category;
            typedef __invalid_type target_type;
        };

        template <>
        struct DeviceTraits<CPU> {
            typedef __host_target target_category;
            typedef __cpu_device target_type;
        };

        template <>
        struct DeviceTraits<ARM> {
            typedef __host_target target_category;
            typedef __arm_device target_type;
        };

        //!data traits
        template <DataType type>
        struct DataTrait{
            typedef __invalid_type dtype;
        };

        template <>
        struct DataTrait<PM_HALF> {
            typedef short dtype;
        };

        template <>
        struct DataTrait<PM_FLOAT> {
            typedef float dtype;
        };

        template <>
        struct DataTrait<PM_DOUBLE> {
            typedef double dtype;
        };

        template <>
        struct DataTrait<PM_INT8> {
            typedef char dtype;
        };

        template <>
        struct DataTrait<PM_INT16> {
            typedef short dtype;
        };

        template <>
        struct DataTrait<PM_INT32> {
            typedef int dtype;
        };

        template <>
        struct DataTrait<PM_INT64> {
            typedef long dtype;
        };

        template <>
        struct DataTrait<PM_UINT8> {
            typedef unsigned char dtype;
        };

        template <>
        struct DataTrait<PM_UINT16> {
            typedef unsigned short dtype;
        };

        template <>
        struct DataTrait<PM_UINT32> {
            typedef unsigned int dtype;
        };


    }
}


