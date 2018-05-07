
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

#include "common/types.h"
using namespace std;

namespace paddle_mobile {
    namespace framework {

#define PM_CHECK(condition) \
    do { \
    PMStatus error = condition; \
    if(error == PMSuccess) { \
 cout<< pm_get_error_string(error); \
        break; } \
} while (0)

inline const char* pm_get_error_string(PMStatus error_code){
    switch (error_code) {
        case PMSuccess:
            return "ANAKIN_SABER_STATUS_SUCCESS";
        case PMNotInitialized:
            return "ANAKIN_SABER_STATUS_NOT_INITIALIZED";
        case PMInvalidValue:
            return "ANAKIN_SABER_STATUS_INVALID_VALUE";
        case PMMemAllocFailed:
            return "ANAKIN_SABER_STATUS_MEMALLOC_FAILED";
        case PMUnKownError:
            return "ANAKIN_SABER_STATUS_UNKNOWN_ERROR";
        case PMOutOfAuthority:
            return "ANAKIN_SABER_STATUS_OUT_OF_AUTHORITH";
        case PMOutOfMem:
            return "ANAKIN_SABER_STATUS_OUT_OF_MEMORY";
        case PMUnImplError:
            return "ANAKIN_SABER_STATUS_UNIMPL_ERROR";
        case PMWrongDevice:
            return "ANAKIN_SABER_STATUS_WRONG_DEVICE";
    }
    return "ANAKIN SABER UNKOWN ERRORS";
}

        template <bool If, typename ThenType, typename ElseType>
        struct IF {
            /// Conditional type result
            typedef ThenType Type;      // true
        };

        template <typename ThenType, typename ElseType>
        struct IF<false, ThenType, ElseType> {
            typedef ElseType Type;      // false
        };

    }
}