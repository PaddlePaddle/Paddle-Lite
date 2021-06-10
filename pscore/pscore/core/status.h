#pragma once

#include <absl/strings/string_view.h>
#include <memory>
#include "pscore/core/pb/error_codes.pb.h"

namespace pscore {

const std::string& EmptyString();

class Status {
 public:
  using code_t = pscore::error::Code;

  Status() = default;

  Status(code_t code, absl::string_view msg);
  Status(const Status& s);

  Status& operator=(const Status& s);

  static Status OK() { return Status(); }

  bool ok() const { return state_ == nullptr; }

  code_t code() const { return ok() ? code_t::OK : state_->code; }
  const std::string& error_message() const {
    return ok() ? EmptyString() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  //! Update the existing Status with the new state \p s, convienient way of
  //! keeping track of the first error encountered.
  void Update(const Status& s);

  std::string ToString() const;

 private:
  struct State {
    pscore::error::Code code;
    std::string msg;
  };

  std::unique_ptr<State> state_;
};

namespace errors {

Status Unimplemented(absl::string_view msg);

}  // namespace errors

}  // namespace pscore
