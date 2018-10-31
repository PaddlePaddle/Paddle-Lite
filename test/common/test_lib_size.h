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

//
// Created by liuRuiLong on 2018/6/6.
//

#ifndef PADDLE_MOBILE_TEST_LIB_SIZE_H
#define PADDLE_MOBILE_TEST_LIB_SIZE_H

#include <pthread.h>
#include <thread>
#include <vector>
//#include <list>
//#include <tuple>
//#include <typeinfo>
//#include <mutex>
//#include <initializer_list>
//#include <map>
//#include <string>
//#include <unordered_map>
//#include <unordered_set>
//#include <algorithm>

//#include <iostream>
//#include <sstream>
//#include <memory>
//#include <stdio.h>
//#include <cstring>

void foo() {
  //  char *str = "1234";
  //  char dst[10];
  //  strcpy(dst, str);

  //  std::cout << "12345" << std::endl;
  std::vector<int> vec = {1, 2, 3, 4, 5};
  vec.push_back(2);

  pthread_mutex_init(NULL, NULL);
  pthread_attr_destroy(NULL);
  //  std::find(vec.begin(), vec.end(), 1);

  //  std::list<int> l;
  //  std::mutex mutex_;

  //  std::map<int, float> m;
  //  std::unordered_map<int, float> u_m;
  //  std::unordered_set<int> u_s;
  //  std::string ss = "12345";
  //  printf("%f", ss.c_str());

  //  std::initializer_list<int> init_list = {1, 2};
  //  std::tuple<int, int> t = {1, 2};

  //  std::tuple_element<I, std::tuple<ARGS...>>::type

  //  std::tuple<>

  //  int i;
  //  int j;
  //  if (typeid(i) == typeid(j)){
  //    int z = 10;
  //  }

  //  std::shared_ptr<int> s1 = std::make_shared<int>();

  //  std::stringstream ss;
  //  ss << "12345";
}

class test_lib_size {
 public:
  test_lib_size() {}
  //  std::shared_ptr<int> Test(){
  //    std::vector<int> vec = {1, 2, 3};
  //    std::shared_ptr<int> si = std::make_shared<int>();
  //    return si;
  //  }

  //  void test(){
  //    int i = 9;
  //  }
};

#endif  // PADDLE_MOBILE_TEST_LIB_SIZE_H
