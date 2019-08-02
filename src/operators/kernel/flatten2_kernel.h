//
// Created by hujie09 on 2019-07-31.
//

#ifdef FLATTEN2_OP
#include <operators/op_param.h>
#include "framework/operator.h"
namespace paddle_mobile {
    namespace operators {
        DECLARE_KERNEL(Flatten2, FlattenParam)
    }
}  // namespace paddle_mobile

#endif //FLATTEN2_KERNEL
