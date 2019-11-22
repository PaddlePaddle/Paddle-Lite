/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "common/util.h"

#ifdef MODEL_SECU
#include <stdlib.h>
#include <stdio.h>
#include "seco/seco.h"  
#include <iostream> 
#endif

namespace paddle_mobile {

char *ReadFileToBuff(std::string filename) {
#ifdef MODEL_SECU

Seco *seco = new Seco();
   unsigned char pub_key[512];
   int value = seco->read_pubkey_from_chip(pub_key);
   if (value!=0){
        delete seco;
        std::cout<<"read chip encrypt key error"<<std::endl;
        return nullptr;
    }
    long out_length = 0;
    unsigned char *out = nullptr;
    value = seco->parse_model(pub_key, filename, &out, out_length);  
    delete seco;
    if (value!=0){
        std::cout<<"parse_model error and error code is:"<<value<<std::endl;
        return nullptr;  
    }
    
    return (char*)out;
#else
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "file should not be empty");
  rewind(file);
  char *data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  PADDLE_MOBILE_ENFORCE(bytes_read == size,
                        "read binary file bytes do not match with fseek");
  fclose(file);
  return data;
#endif
}

}  // namespace paddle_mobile
