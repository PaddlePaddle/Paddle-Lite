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

#include <vector>

#include "lod_tensor.h"
#include "tensor.h"

namespace paddle_mobile {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
  }

  SelectedRows() {
    height_ = 0;
    value_.reset(new Tensor());
  }

  const Tensor& value() const { return *value_; }

  Tensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const std::vector<int64_t>& rows() const { return rows_; }

  std::vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const std::vector<int64_t>& rows) { rows_ = rows; }

  /**
   * get the index of id in rows
   */
  int64_t index(int64_t id) const {
    auto it = std::find(rows_.begin(), rows_.end(), id);
    //    PADDLE_ENFORCE(it != rows_.end(), "id should be in rows");
    return static_cast<int64_t>(std::distance(rows_.begin(), it));
  }

  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRows are simply concated when adding together. Until a
  // SelectedRows add a Tensor, will the duplicate rows be handled.
  std::vector<int64_t> rows_;
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;
};

}  // namespace framework
}  // namespace paddle_mobile