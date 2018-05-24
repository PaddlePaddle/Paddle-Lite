#pragma once

#include <vector>
#ifdef PADDLE_MOBILE_DEBUG
#include <iostream>
#include <sstream>
#include <string>
#endif

namespace paddle_mobile {

#ifdef PADDLE_MOBILE_DEBUG

enum LogLevel {
  kNO_LOG,
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4
};

// log level
static LogLevel log_level = kLOG_DEBUG4;

static std::vector<std::string> logs{"NO",      "ERROR ",  "WARNING",
                                     "INFO   ", "DEBUG  ", "DEBUG1 ",
                                     "DEBUG2 ", "DEBUG3 ", "DEBUG4 "};
struct ToLog;
struct Print;

struct Print {
  friend struct ToLog;

  template <typename T>
  Print &operator<<(T const &value) {
    buffer_ << value;
    return *this;
  }

 private:
  void print(LogLevel level) {
    buffer_ << std::endl;
    if (level == kLOG_ERROR) {
      std::cerr << buffer_.str();
    } else {
      std::cout << buffer_.str();
    }
  }
  std::ostringstream buffer_;
};

struct ToLog {
  ToLog(LogLevel level = kLOG_DEBUG, const std::string &info = "")
      : level_(level) {
    unsigned blanks =
        (unsigned)(level > kLOG_DEBUG ? (level - kLOG_DEBUG) * 4 : 1);
    printer_ << logs[level] << " " << info << ":" << std::string(blanks, ' ');
  }

  template <typename T>
  ToLog &operator<<(T const &value) {
    printer_ << value;
    return *this;
  }

  ~ToLog() { printer_.print(level_); }

 private:
  LogLevel level_;
  Print printer_;
};

#define LOG(level)                                                             \
  if (level > paddle_mobile::log_level) {                                      \
  } else                                                                       \
    paddle_mobile::ToLog(                                                      \
        level,                                                                 \
        (std::stringstream()                                                   \
         << "[file: "                                                          \
         << (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__) \
         << "] [line: " << __LINE__ << "] ")                                   \
            .str())

#define DLOG                                                                   \
  if (paddle_mobile::kLOG_DEBUG > paddle_mobile::log_level) {                  \
  } else                                                                       \
    paddle_mobile::ToLog(                                                      \
        paddle_mobile::kLOG_DEBUG,                                             \
        (std::stringstream()                                                   \
         << "[file: "                                                          \
         << (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__) \
         << "] [line: " << __LINE__ << "] ")                                   \
            .str())

#define LOGF(level, format, ...)          \
  if (level > paddle_mobile::log_level) { \
  } else                                  \
    printf(format, ##__VA_ARGS__)

#define DLOGF(format, ...)                                    \
  if (paddle_mobile::kLOG_DEBUG > paddle_mobile::log_level) { \
  } else                                                      \
    printf(format, ##__VA_ARGS__)

#else

enum LogLevel {
  kNO_LOG,
  kLOG_ERROR,
  kLOG_WARNING,
  kLOG_INFO,
  kLOG_DEBUG,
  kLOG_DEBUG1,
  kLOG_DEBUG2,
  kLOG_DEBUG3,
  kLOG_DEBUG4
};

struct ToLog;
struct Print {
  friend struct ToLog;
  template <typename T>
  Print &operator<<(T const &value) {}

 private:
};

struct ToLog {
  ToLog(LogLevel level) {}

  template <typename T>
  ToLog &operator<<(T const &value) {
    return *this;
  }
};

#define LOG(level) \
  if (true) {      \
  } else           \
    paddle_mobile::ToLog(level)

#define DLOG  \
  if (true) { \
  } else      \
    paddle_mobile::ToLog(paddle_mobile::kLOG_DEBUG)

#define LOGF(level, format, ...)

#define DLOGF(format, ...)

#endif

template <typename T>
Print &operator<<(Print &printer, const std::vector<T> &v) {
  printer << "[ ";
  for (const auto &value : v) {
    printer << value << " ";
  }
  printer << " ]";
  return printer;
}

}  // namespace paddle_mobile