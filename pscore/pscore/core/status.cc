#include "pscore/core/status.h"
#include "pscore/common/macros.h"

namespace pscore {

const std::string &EmptyString() {
  static std::unique_ptr<std::string> x(new std::string());
  return *x;
}

absl::string_view GetErrorCodeName(Status::code_t code) {
  switch (code) {
    case Status::code_t::OK:
      return "OK";
    case Status::code_t::CANCELLED:
      return "CANCELED";
    default:
      PSCORE_NOT_IMPLEMENTED
      return "";
  }
}

Status::Status(Status::code_t code, absl::string_view msg) {
  if (code != code_t::OK) {
    state_.reset(new State{code, std::string(msg)});
  }
}

Status::Status(const Status &s) { *this = s; }

Status &Status::operator=(const Status &s) {
  if (s.ok()) {
    state_.reset();
  } else {
    if (state_) {
      state_->code = s.code();
      state_->msg = s.error_message();
    } else {
      state_.reset(new State{s.code(), s.error_message()});
    }
  }
  return *this;
}

bool Status::operator==(const Status &x) const {
  if (ok()) return x.ok();
  if (x.ok()) return ok();

  return code() == x.code() && error_message() == x.error_message();
}

bool Status::operator!=(const Status &x) const { return !(*this == x); }

void Status::Update(const Status &s) {
  if (s.ok()) {
    state_.reset();
    return;
  }

  if (ok()) {
    *this = s;
    return;
  }
}

std::string Status::ToString() const {
  if (ok()) return "OK";
  return std::string(GetErrorCodeName(code())) + " " + error_message();
}

namespace errors {

Status Unimplemented(absl::string_view msg) {
  return Status(Status::code_t::NOT_IMPLEMENTED, msg);
}

}  // namespace errors

}  // namespace pscore
