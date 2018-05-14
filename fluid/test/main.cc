#include "io.h"

int main() {
  paddle_mobile::Load(
      std::string("models/image_classification_resnet.inference.model/"));
  return 0;
}
