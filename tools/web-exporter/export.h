#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <ostream>
#include <fstream>

#include "framework/loader.h"
#include "framework/executor.h"
#include "framework/scope.h"
#include "framework/program/program_desc.h"

// using paddle_mobile::framework::ProgramDesc;
// using paddle_mobile::framework::Scope;

using ProgramPtr = std::shared_ptr<paddle_mobile::framework::ProgramDesc>;
using ScopePtr = std::shared_ptr<paddle_mobile::framework::Scope>;

void export_nodejs(ProgramPtr program, ScopePtr scope, std::ostream & os = std::cout);
void export_scope(ProgramPtr program, ScopePtr scope, const std::string & dirname = ".");


template <typename T>
inline std::string var2str(const T & v) {
  return std::to_string(v);
}

template <>
inline std::string var2str(const std::string & v) {
  return "\"" + v + "\"";
}

inline std::string var2str(const char* v) {
  return var2str<std::string>(v);
}

inline std::string var2str(const bool v) {
  return v ? "true" : "false";
}

template <typename T>
std::string var2str(const std::vector<T> & v) {
  std::string r = "[";
  auto s = v.size();
  for (int i = 0; i < s; i++) {
    if (i) r += ", ";
    r += var2str(v[i]);
  }
  return r + "]";
}

struct VarVisitor {
  using type_t = decltype(var2str(0));

  template <typename T>
  type_t operator()(const T & v) {
    return var2str(v);
  }
};