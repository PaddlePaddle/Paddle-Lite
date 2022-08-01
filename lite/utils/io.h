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

#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#if defined(_MSC_VER) or defined(LITE_WITH_QNX)
#include "dirent.h"  // NOLINT
#else
#include <dirent.h>
#endif
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

static bool IsFileExists(const std::string& path) {
  std::ifstream file(path);
  bool res = file.is_open();
  if (res) {
    file.close();
  }
  return res;
}

// ARM mobile not support mkdir in C++
static void MkDirRecur(const std::string& path) {
  if (IsFileExists(path)) {
    LOG(INFO) << "Directory [" << path << "] exists!";
    return;
  }
#ifndef LITE_WITH_ARM

#ifdef _WIN32
  if (system(string_format("md %s", path.c_str()).c_str()) != 0) {
    LOG(ERROR) << "Cann't mkdir " << path;
  }
#else
  if (system(string_format("mkdir -p %s", path.c_str()).c_str()) != 0) {
    LOG(ERROR) << "Cann't mkdir " << path;
  }
#endif  // _WIN32
#else   // On ARM
  CHECK_NE(mkdir(path.c_str(), S_IRWXU), -1) << "Cann't mkdir " << path;
#endif
}

// read buffer from file
static std::string ReadFile(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    LOG(FATAL) << "Open file: [" << filename << "] failed.";
  }
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch)) buf.put(ch);
  ifile.close();
  return buf.str();
}

// read lines from file
static std::vector<std::string> ReadLines(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    LOG(FATAL) << "Open file: [" << filename << "] failed.";
  }
  std::vector<std::string> res;
  std::string tmp;
  while (getline(ifile, tmp)) res.push_back(tmp);
  ifile.close();
  return res;
}

static void WriteLines(const std::vector<std::string>& lines,
                       const std::string& filename) {
  std::ofstream ofile(filename.c_str());
  if (!ofile.is_open()) {
    LOG(FATAL) << "Open file: [" << filename << "] failed.";
  }
  for (const auto& line : lines) {
    ofile << line << "\n";
  }
  ofile.close();
}

static bool IsDir(const std::string& path) {
  DIR* dir_fd = opendir(path.c_str());
  if (dir_fd == nullptr) return false;
  closedir(dir_fd);
  return true;
}

static std::vector<std::string> ListDir(const std::string& path,
                                        bool only_dir = false) {
  if (!IsDir(path)) {
    LOG(FATAL) << "[" << path << "] is not a valid dir path.";
  }

  std::vector<std::string> paths;
  DIR* parent_dir_fd = opendir(path.c_str());
  dirent* dp;
  while ((dp = readdir(parent_dir_fd)) != nullptr) {
    // Exclude '.', '..' and hidden dir
    std::string name(dp->d_name);
    if (name == "." || name == ".." || name[0] == '.') continue;
    if (IsDir(Join<std::string>({path, name}, "/"))) {
      paths.push_back(name);
    }
  }
  closedir(parent_dir_fd);
  return paths;
}

static std::vector<std::string> ListFile(const std::string& path) {
  if (!IsDir(path)) {
    LOG(FATAL) << "[" << path << "] is not a valid dir path.";
  }

  std::vector<std::string> paths;
  DIR* parent_dir_fd = opendir(path.c_str());
  dirent* dp;
  while ((dp = readdir(parent_dir_fd)) != nullptr) {
    // Exclude '.', '..' and hidden dir
    std::string name(dp->d_name);
    if (name == "." || name == ".." || name[0] == '.') continue;
// Is file
#ifndef LITE_WITH_QNX
    if (dp->d_type == DT_REG) {
      paths.push_back(name);
    }
#endif
  }
  closedir(parent_dir_fd);
  return paths;
}

static bool ReadFile(const std::string& filename,
                     std::vector<char>* contents,
                     bool binary = true) {
  FILE* fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char* ptr = reinterpret_cast<char*>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

template <typename T>
static bool ReadFile(const std::string& filename, std::vector<T>* contents) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp) / sizeof(T);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  T* ptr = reinterpret_cast<T*>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, sizeof(T), size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

static bool WriteFile(const std::string& filename,
                      const std::vector<char>& contents,
                      bool binary = true) {
  FILE* fp = fopen(filename.c_str(), binary ? "wb" : "w");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const char* ptr = reinterpret_cast<const char*>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

template <typename T>
static bool WriteFile(const std::string& filename,
                      const std::vector<T>& contents) {
  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const T* ptr = reinterpret_cast<const T*>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, sizeof(T), size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

}  // namespace lite
}  // namespace paddle
