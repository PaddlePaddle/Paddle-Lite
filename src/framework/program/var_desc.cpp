#include "var_desc.h"

namespace paddle_mobile {

namespace framework {

VarDesc::VarDesc(const proto::VarDesc &desc) : desc_(desc) {}

}  // namespace framework
}  // namespace paddle_mobile
