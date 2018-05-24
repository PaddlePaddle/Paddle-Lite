#include "attribute.h"

namespace paddle_mobile {
namespace framework {



/*
 * Variant<int, float, std::string, std::vector<int>, std::vector<float>,
          std::vector<std::string>, bool, std::vector<bool>, BlockDesc *,
          int64_t>
 * */

struct PrintVistor: Vistor<Print &>{
  PrintVistor(Print &printer):printer_(printer){
  }
  template <typename T>
  Print &operator()(const T &value){
    printer_ << value;
    return printer_;
  }
 private:
  Print &printer_;
};

Print &operator<<(Print &printer, const Attribute &attr) {
  Attribute::ApplyVistor(PrintVistor(printer), attr);
//  std::vector<std::string> v = {"1", "2"};
//  printer << (v);
  return printer;
}

}
}  // namespace paddle_mobile
