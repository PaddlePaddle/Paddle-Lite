// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "lite/tests/utils/unit_test/engine_test.h"
#include "lite/tests/utils/unit_test/test_base.h"

/**
 * \brief declare the Test Function
 */
/*#define TEST(test_class, test_function)	\
  class test_class##_##test_function:public test_class{\
  public:\
    friend class ::lite::test::EnginResOp;\
    void test_function();\
  };\
  const test_class##_##test_function _##test_class##_##test_function;\
  {\
    ::lite::test::EnginResOp::GetInstance(#test_class,#test_function)>>test_class::GetInstance()&
  \
    std::bind(&test_class##_##test_function::test_function,&_test_class##_##test_function);\
  } \
  void test_class##_##test_function::test_function()
*/
#define TEST(test_class, test_function)                               \
  class test_class##_##test_function : public test_class {            \
   public:                                                            \
    friend class ::lite::test::EnginResOp;                            \
    void test_function();                                             \
  };                                                                  \
  const test_class##_##test_function _##test_class##_##test_function; \
  std::function<void(void)> func_##test_class##_##test_function =     \
      std::bind(&test_class##_##test_function::test_function,         \
                _##test_class##_##test_function);                     \
  ::lite::test::EnginResOp op_test_class##_##test_function =          \
      (::lite::test::EnginResOp(#test_class, #test_function) >>       \
           test_class::get_instance<test_class>() &                   \
       func_##test_class##_##test_function);                          \
  void test_class##_##test_function::test_function()

#define InitTest() ::lite::test::config::initial()

#define RUN_ALL_TESTS(argv_0) \
  ::lite::test::EngineTest::get_instance().run_all(argv_0)
