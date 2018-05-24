#pragma once

#include <cctype>
#include <iostream>
#include <string>

namespace paddle_mobile {
namespace framework {

enum class DataLayout {
  kNHWC = 0,
  kNCHW = 1,
  kAnyLayout = 2,
};

inline DataLayout StringToDataLayout(const std::string &str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return DataLayout::kAnyLayout;
  } else {
    //    std::cout << "Unknown storage order string: %s", s;
  }
}

inline std::string DataLayoutToString(const DataLayout &data_layout) {
  switch (data_layout) {
    case DataLayout::kNHWC:
      return "NHWC";
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kAnyLayout:
      return "ANY_LAYOUT";
    default:
      break;
      //      std::cout << "unknown DataLayou %d", data_layout;
  }
}

inline std::ostream &operator<<(std::ostream &out, const DataLayout &l) {
  out << DataLayoutToString(l);
  return out;
}

}  // namespace framework
}  // namespace paddle_mobile
