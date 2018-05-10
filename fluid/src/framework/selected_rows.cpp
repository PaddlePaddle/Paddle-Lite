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

#include "selected_rows.h"

namespace paddle_mobile {
namespace framework {

//    void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
//                           const platform::DeviceContext& dev_ctx) {
//        {  // the 1st field, uint32_t version
//            constexpr uint32_t version = 0;
//            os.write(reinterpret_cast<const char*>(&version), sizeof(version));
//        }
//        {
//            // the 2st field, rows information
//            auto& rows = selected_rows.rows();
//            uint64_t size = rows.size();
//            os.write(reinterpret_cast<const char*>(&size), sizeof(size));
//            for (uint64_t i = 0; i < size; ++i) {
//                os.write(reinterpret_cast<const char*>(&rows[i]), sizeof(rows[i]));
//            }
//        }
//        {
//            // the 3st field, the height of SelectedRows
//            int64_t height = selected_rows.height();
//            os.write(reinterpret_cast<const char*>(&height), sizeof(height));
//        }
//        // the 4st field, Tensor data
//        TensorToStream(os, selected_rows.value(), dev_ctx);
//    }

//    void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
//                               const platform::DeviceContext& dev_ctx) {
//        {
//            // the 1st field, unit32_t version for SelectedRows
//            uint32_t version;
//            is.read(reinterpret_cast<char*>(&version), sizeof(version));
////            PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
//        }
//        {
//            // the 2st field, rows information
//            uint64_t size;
//            is.read(reinterpret_cast<char*>(&size), sizeof(size));
//            auto& rows = *selected_rows->mutable_rows();
//            rows.resize(size);
//            for (uint64_t i = 0; i < size; ++i) {
//                is.read(reinterpret_cast<char*>(&rows[i]), sizeof(int64_t));
//            }
//        }
//        {
//            // the 3st field, the height of the SelectedRows
//            int64_t height;
//            is.read(reinterpret_cast<char*>(&height), sizeof(int64_t));
//            selected_rows->set_height(height);
//        }
//        // the 4st field, tensor which contains the data
//        TensorFromStream(is, selected_rows->mutable_value(), dev_ctx);
//    }

}  // namespace framework
}  // namespace paddle