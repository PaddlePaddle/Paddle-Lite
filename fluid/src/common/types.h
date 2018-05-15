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

#pragma once;

namespace paddle_mobile {
enum class Precision : int { FP32 = 0 };

//! device type
enum DeviceTypeEnum { kINVALID = -1, kCPU = 0, kFPGA = 1, kGPU_MALI = 2 };

template <DeviceTypeEnum T>
struct DeviceType {};

typedef DeviceType<kCPU> CPU;
typedef DeviceType<kFPGA> FPGA;
typedef DeviceType<kGPU_MALI> GPU_MALI;

//! data type
enum DataType {
  PM_INVALID = -1,
  PM_HALF = 0,
  PM_FLOAT = 1,
  PM_DOUBLE = 2,
  PM_INT8 = 3,
  PM_INT16 = 4,
  PM_INT32 = 5,
  PM_INT64 = 6,
  PM_UINT8 = 7,
  PM_UINT16 = 8,
  PM_UINT32 = 9,
  PM_STRING = 10,
  PM_BOOL = 11,
  PM_SHAPE = 12,
  PM_TENSOR = 13
};
//!
enum PMStatus{
  PMSuccess = 0xFF,       /*!< No errors */
  PMNotInitialized = 0x01, /*!< Data not initialized. */
  PMInvalidValue = 0x02, /*!< Incorrect variable value. */
  PMMemAllocFailed = 0x03, /*!< Memory allocation error. */
  PMUnKownError = 0x04,  /*!< Unknown error. */
  PMOutOfAuthority = 0x05, /*!< Try to modified data not your own*/
  PMOutOfMem = 0x06, /*!< OOM error*/
  PMUnImplError = 0x07,    /*!< Unimplement error. */
  PMWrongDevice = 0x08  /*!< un-correct device. */
} ;
}
