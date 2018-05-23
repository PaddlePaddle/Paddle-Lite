#include "common/log.h"

int main() {

  DLOGF("DASJFDAFJ%d -- %f", 12345, 344.234);

  LOGF(paddle_mobile::kLOG_DEBUG, "DASJFDAFJ%d -- %f", 12345, 344.234);

  LOG(paddle_mobile::kLOG_DEBUG) << "test debug"
                                 << " next log";

  LOG(paddle_mobile::kLOG_DEBUG1) << "test debug1"
                                  << " next log";
  LOG(paddle_mobile::kLOG_DEBUG2) << "test debug2"
                                  << " next log";
  DLOG << "test DLOG";

  LOG(paddle_mobile::kLOG_ERROR) << " error occur !";

  return 0;
}
