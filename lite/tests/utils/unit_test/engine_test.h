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

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/tests/utils/unit_test/test_base.h"
namespace lite {
namespace test {

class EngineTest;

class EnginRes {
 public:
  static EnginRes& get_instance() {
    static EnginRes ins;
    return ins;
  }

  friend class EngineTest;
  friend class EnginResOp;

 private:
  /**
  * \brief store the test class  and it's  callback function.
  */
  std::unordered_map<Test*, std::vector<std::function<void(void)>>>
      class2func_map_;
  /**
  * \brief store the test class  and it's  callback name.
  */
  std::unordered_map<std::string, std::vector<std::string>> class2funcname_map_;
  /**
  * \brief store the test class  and it's  class name.
  */
  std::unordered_map<std::string, Test*> name2class_map_;
};

class EnginResOp {
 public:
  ~EnginResOp() {}
  EnginResOp(const char* class_name, const char* func_name) {
    _test_class_name = class_name;
    _test_func_name = func_name;
    EnginRes& ngtest_res = EnginRes::get_instance();
    ngtest_res.class2funcname_map_[class_name].push_back(func_name);
    _func_num = ngtest_res.class2funcname_map_[class_name].size();
  }

  /*inline static EnginResOp& GetInstance(const char* className, const char*
  funcName){
      static EnginResOp ins(const char* className, const char* funcName);
      return ins;
  }*/

  EnginResOp& operator>>(Test* test_class) {
    EnginRes& ngtest_res = EnginRes::get_instance();
    _test_class = test_class;
    ngtest_res.class2func_map_[test_class].resize(_func_num + 1);
    ngtest_res.name2class_map_[_test_class_name] = test_class;
    return *this;
  }

  EnginResOp& operator&(const std::function<void(void)>& test_func) {
    EnginRes& ngtest_res = EnginRes::get_instance();
    ngtest_res.class2func_map_[_test_class][_func_num - 1] = test_func;
    return *this;
  }

  // EnginResOp(const EnginResOp&) = delete;
  // EnginResOp& operator=(const EnginResOp&) = delete;
 private:
  Test* _test_class;
  std::string _test_class_name;
  std::string _test_func_name;
  int _func_num;
};

class EngineTest {
 public:
  EngineTest() {}

  /**
  * \brief get singleton instance of test engine.
  */

  inline static EngineTest& get_instance() {
    static EngineTest ins;
    return ins;
  }

  /**
  * \brief run all test.
  */
  bool run_all(const char* app_name) {
    fprintf(stderr,
            "%s%s%s[***********]%s %sRunning main() for %s%s.\n",
            reset(),
            bold(),
            green(),
            reset(),
            dim(),
            app_name,
            reset());
    EnginRes& ngtest_res = EnginRes::get_instance();
    int test_case_num = 0, test_func_num = 0;
    for (auto& i : ngtest_res.class2funcname_map_) {
      ++test_case_num;
      test_func_num += i.second.size();
    }
    fprintf(stderr,
            "%s%s%s[    SUM    ]%s %sRunning%s %s%d%s %stest function from%s "
            "%s%d%s %stest class%s.\n\n",
            reset(),
            bold(),
            green(),
            reset(),
            dim(),
            reset(),
            white(),
            test_func_num,
            reset(),
            dim(),
            reset(),
            white(),
            test_case_num,
            reset(),
            dim(),
            reset());
    if (test_case_num == 0 || test_func_num == 0) {
      return false;
    }
    for (auto pair : ngtest_res.class2funcname_map_) {
      std::string test_class_name = pair.first;
      std::vector<std::string> testFuncNameVec = pair.second;
      fprintf(stderr,
              "%s%s%s[===========]%s Running %d tests from %s.\n",
              reset(),
              bold(),
              green(),
              reset(),
              static_cast<int>(testFuncNameVec.size()),
              test_class_name.c_str());
      double sum = 0.0;
      for (size_t i = 0; i < testFuncNameVec.size(); i++) {
        fprintf(stderr,
                "%s%s[ RUN       ]%s %s.%s\n",
                reset(),
                green(),
                reset(),
                test_class_name.c_str(),
                testFuncNameVec[i].c_str());
        Counter elapsedT;
        elapsedT.start();
        // invoke the test function
        std::function<void(void)> test_func =
            ngtest_res
                .class2func_map_[ngtest_res.name2class_map_[test_class_name]]
                                [i];
        test_func();
        elapsedT.end();
        fprintf(stderr,
                "%s%s[        OK ]%s %s.%s (%s %0.2lf ms%s )\n",
                reset(),
                green(),
                reset(),
                test_class_name.c_str(),
                testFuncNameVec[i].c_str(),
                red(),
                elapsedT.elapsed_time(),
                reset());
        sum += elapsedT.elapsed_time();
      }
      fprintf(stderr,
              "%s%s%s[===========]%s %d tests from %s class ran. ( %s%0.2lf "
              "ms%s total )\n\n",
              reset(),
              bold(),
              green(),
              reset(),
              static_cast<int>(testFuncNameVec.size()),
              test_class_name.c_str(),
              red(),
              sum,
              reset());
    }
    return true;
  }

  EngineTest(const EngineTest&) = delete;             // disable copy construct
  EngineTest& operator=(const EngineTest&) = delete;  // disable copy operator
};

}  // namespace test
}  // namespace lite
