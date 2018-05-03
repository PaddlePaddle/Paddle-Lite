
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
#include "paddle_mobile_object.h"
namespace paddle_mobile{
    namespace framework{

        struct __invalid_type{};

        //! device type
        enum DeviceTypeEnum{
            eINVALID = -1,
            eARM = 0,
            eCPU = 1,
        };

        template <DeviceTypeEnum T>
        struct DeviceType{};

        typedef DeviceType<eARM> ARM;
        typedef DeviceType<eCPU> CPU;

        //! data_layout type
        struct W{};
        struct HW{};
        struct WH{};
        struct NW{};
        struct NHW{};
        struct NCHW{};
        struct NHWC{};
        struct NCHW_C4{};

        //! dim type
        struct D5{};
        struct D4{};
        struct D3{};
        struct D2{};
        struct D1{};

        //! data type
        enum DataType {
            PM_INVALID      =       -1,
            PM_HALF         =       0,
            PM_FLOAT        =       1,
            PM_DOUBLE       =       2,
            PM_INT8         =       3,
            PM_INT16        =       4,
            PM_INT32        =       5,
            PM_INT64        =       6,
            PM_UINT8        =       7,
            PM_UINT16       =       8,
            PM_UINT32       =       9,
            PM_STRING       =       10,
            PM_BOOL         =       11,
            PM_SHAPE        =       12,
            PM_TENSOR       =       13
        };
        //!
        typedef enum {
            PMSuccess         = -1,                             /*!< No errors */
            PMNotInitialized  = 1,                              /*!< Data not initialized. */
            PMInvalidValue    = (1 << 1) + PMNotInitialized, /*!< Incorrect variable value. */
            PMMemAllocFailed  = (1 << 2) + PMInvalidValue,   /*!< Memory allocation error. */
            PMUnKownError     = (1 << 3) + PMMemAllocFailed, /*!< Unknown error. */
            PMOutOfAuthority  = (1 << 4) + PMUnKownError,    /*!< Try to modified data not your own*/
            PMOutOfMem        = (1 << 5) + PMOutOfAuthority, /*!< OOM error*/
            PMUnImplError     = (1 << 6) + PMOutOfMem,       /*!< Unimplement error. */
            PMWrongDevice     = (1 << 7) + PMUnImplError     /*!< un-correct device. */
        } PMStatus;

    }
}