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
// This file contains the file system of the lite system. Every data type in
// Variable should be registered here, and the analysis phase will check the
// data type correction.
// This mechanism is made for keeping our system simpler and more stable, for
// the dubious typed Variables in the Operators' inputs and outputs are disaster
// for analysis and runtime.

#include <map>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include "lite/core/kernel_version.h"
#include "lite/core/tensor.h"
#include "lite/core/version.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {

// Type is the definition of all the types that supported by the Variable that
// represents as the input and output of an operator or kernel.
// The DNN system is simple, just a list of operators, and the architecture
// can not process that many data types as a compiler, or that will turn out to
// a chaos.
//
// We should make sure that the supported data types be registered here, and
// keep the set small and avoid using some special data types as op's
// inputs or outputs, such as some runtime cache, those types can't be processed
// by the MIR.
//
// A tensor with different places(target, precision, data layout or device)
// should be treated as different types. Different types might be compatible
// with each other, for example, the `VoidTy` means any type, so any other types
// can be treated as a `VoidTy`.
//
// The Different Types can transform to others by adding some special
// transforming operators, for example, a DataLayoutTransformOp can convert a
// `TensorFp32NCHWTy` to a `TensorFp32NHWCTy`; a IoCopyOp can convert a
// `TensorFp32NCHWTy(kHost)` to `TensorFp32NCHWTy(kCUDA)`. There are many other
// convertions between different Types, but there are some unsupported type
// convertions, for example, there is noway to convert a `UnsupportedTy` to a
// `TensorAnyTy`.
//
// We use Types to declare the definition of a kernel, each inputs' and outputs'
// arguments have a specific Types.
//
// REGISTER_LITE_KERNEL(mul, kARM, kInt8, kNCHW, Mul_int8_f32, def)
//     .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
//     .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
//     .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
//     .Finalize();
//
// The above definition will be used in MIR by Type inference and uncompatible
// types check.
//
// TODO(Superjomn) Add operator/kernel-wise static checking to avoid unsupported
// type mixed in the system.
class DataType {
 public:
  // The Void type can cast to any other type.
  // The Unsupported is the data type that developed include in the system, for
  // example, some `std::set` is used as input of some operator. It wan't be
  // analyzed or optimized by the system, that way results in many bugs in
  // previous system, so it should be avoided.
  enum class ID : int {
    Void = 0,     // unknown type that can be cast to any data type.
    Unsupported,  // Unsupported data type that will not be analyzed.
    // Tensor_Any represents a Tensor with any place, data, layout. It is used
    // in some IO kernels those doesn't care the data.
    Tensor,
    // A tensor list, but all the elements should have the same type.
    TensorList,
    // A vector of local scope, which size equals the step number of While Op.
    // The i'th scope storages temporary variables generated in the i'th step.
    StepScope,
    // ---------
    NumTypes,  // Must remains as last defined ID.
  };

  ID id() const { return id_; }

  // type check.
  bool IsVoid() const { return id_ == ID::Void; }
  bool IsUnsupported() const { return id_ == ID::Unsupported; }
  bool IsTensor() const { return id_ == ID::Tensor; }
  bool IsTensorList() const { return id_ == ID::TensorList; }
  bool IsStepScope() const { return id_ == ID::StepScope; }
  // Get number of types.
  int num_types() const { return static_cast<int>(ID::NumTypes); }

 protected:
  // Can only extended by subclass.
  explicit DataType(ID id) : id_(id) {}

  ID id_{ID::Unsupported};
};

/*
 * Datatype with Place info considered.
 * NOTE A Type with different Place info is treated as different Type.
 */
class Type : public DataType {
 public:
  // Can cast to another type. This is heavily used in MIR, by determine whether
  // is possible to add a statement to transform a type to another.
  virtual bool TypeCastable(const Type& type) const { return id_ == type.id(); }

  /// Get a Tensor type.
  static const Type* GetTensorTy(TargetType target,
                                 PrecisionType precision = PRECISION(kFloat),
                                 DataLayoutType layout = DATALAYOUT(kNCHW),
                                 int device = 0);
  /// Get a TensorList type.
  static const Type* GetTensorListTy(
      TargetType target,
      PrecisionType precision = PRECISION(kFloat),
      DataLayoutType layout = DATALAYOUT(kNCHW),
      int device = 0);
  /// Get a StepScope type.
  static const Type* GetStepScopeTy();
  /// Get an Unsupported type.
  static const Type* GetUnsupportedTy();
  /// Get an Void type.
  static const Type* GetVoidTy();

  static const Type* Get(DataType::ID type_id,
                         TargetType target = TARGET(kUnk),
                         PrecisionType precision = PRECISION(kUnk),
                         DataLayoutType layout = DATALAYOUT(kUnk),
                         int device = 0);

  TargetType target() const { return place_.target; }
  PrecisionType precision() const { return place_.precision; }
  DataLayoutType layout() const { return place_.layout; }
  int16_t device() const { return place().device; }
  const Place& place() const { return place_; }
  const std::string& name() const { return name_; }

  bool operator==(const Type& other) {
    return id_ == other.id() && place_ == other.place();
  }
  friend STL::ostream& operator<<(STL::ostream& os, const Type& other);

  virtual ~Type() = default;

 protected:
  /// One should avoid using this construct.
  Type(ID id,
       const std::string& name,
       TargetType target = TargetType::kHost,
       PrecisionType precision = PrecisionType::kFloat,
       DataLayoutType layout = DataLayoutType::kNCHW,
       int16_t device = 0)
      : DataType(id), place_{target, precision, layout, device}, name_(name) {}

  Place place_;
  const std::string name_;
};

// -------------------------------- compatible check ---------------------------
static bool TargetCompatibleTo(const Type& a, const Type& b) {
  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM) ||
           x == TARGET(kAny);
  };
  if (a.IsTensor() || b.IsTensor() || a.IsTensorList() || b.IsTensorList()) {
    return is_host(a.target()) ? is_host(b.target()) : a.target() == b.target();
  }
  return true;
}

static bool DataLayoutCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||                 //
         (a.layout() == b.layout() ||  //
          ((b.layout() == DATALAYOUT(kAny)) &&
           (a.layout() != DATALAYOUT(kImageDefault) &&
            a.layout() != DATALAYOUT(kImageFolder))));
}
static bool DataLayoutCompatible(const Type& a, const Type& b) {
  return a.IsVoid() || b.IsVoid() ||   //
         (a.layout() == b.layout() ||  //
          ((b.layout() == DATALAYOUT(kAny)) &&
           (a.layout() != DATALAYOUT(kImageDefault) &&
            a.layout() != DATALAYOUT(kImageFolder))) ||
          ((a.layout() == DATALAYOUT(kAny)) &&
           (b.layout() != DATALAYOUT(kImageDefault) &&
            b.layout() != DATALAYOUT(kImageFolder))));
}

static bool PrecisionCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||  //
         (((a.IsTensor() && b.IsTensor()) ||
           (a.IsTensorList() && b.IsTensor()) ||
           (a.IsTensor() && b.IsTensorList()) ||
           (a.IsTensorList() && b.IsTensorList())) &&
          (a.precision() == b.precision() ||  //
           b.precision() == PRECISION(kAny) ||
           a.precision() == PRECISION(kAny)));
}
static bool PrecisionCompatible(const Type& a, const Type& b) {
  return a.IsVoid() || b.IsVoid() ||  //
         (((a.IsTensor() && b.IsTensor()) ||
           (a.IsTensorList() && b.IsTensorList())) &&
          (a.precision() == b.precision() ||  //
           b.precision() == PRECISION(kAny) ||
           a.precision() == PRECISION(kAny)));
}

static bool DeviceCompatibleTo(const Type& a, const Type& b) {
  return a.IsVoid() ||  //
         (((a.IsTensor() && b.IsTensor()) ||
           (a.IsTensorList() && b.IsTensorList())) &&  //
          (a.device() == b.device()));
}

// Can type 'a' be passed to 'b' directly.
static bool TypeCompatibleTo(const Type& a, const Type& b) {
  return TargetCompatibleTo(a, b) && DataLayoutCompatibleTo(a, b) &&
         PrecisionCompatibleTo(a, b) && DeviceCompatibleTo(a, b);
}
static bool TypeCompatible(const Type& a, const Type& b) {
  return TargetCompatibleTo(a, b) && DataLayoutCompatible(a, b) &&
         PrecisionCompatible(a, b) && DeviceCompatibleTo(a, b);
}

/*
 * ParamType is used to represent a data type of a parameter for the kernel. It
 * can represent any Variable data type.
 * The element_type_hash is the hash code of the element, it should be
 * registered in the `TypeSystem`.
 */
struct ParamType {
  const Type* type;

  ParamType() = default;
  ParamType(const Type* type) : type(type) {}  // NOLINT

  std::string DebugString() const { return type->name(); }
};

/*
 * The ParamTypeRegistry help register the input and output data types for all
 * the kernels. It is made singleton so that all the objects of the same kernel
 * can share the same information.
 *
 * Usage:
 * for register a kernel for FC operator.
 * ParamTypeRegistry::Global().Register(
 *        "fc", {TARGET(kCUDA), PRECISION(kFloat)}, 0,
 *        {typeid(Tensor), {TARGET(kCUDA)}});
 */
class ParamTypeRegistry {
 public:
  enum class IO : int { kInvalid = 0, kInput, kOutput };
  /*
   * Helper class for registering a ParamType for a Kernel.
   * Usage:
   *
   * NewInstance<TARGET(kHost), PRECISION(kFloat)>("fc")
   *   .BindInput("Input_0", {Type::GetTensorTy(TARGET(kHost),
   * PRECISION(kInt64))})
   *   .BindInput("Input_1", {Type::GetTensorTy(TARGET(kHost),
   * PRECISION(kInt64))});
   */
  template <TargetType target,
            PrecisionType precision,
            DataLayoutType layout = DataLayoutType::kNCHW>
  struct NewInstance {
    explicit NewInstance(const std::string& kernel_type)
        : kernel_type_(kernel_type) {}

    NewInstance& BindInput(const std::string& arg_name,
                           const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kInput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }
    NewInstance& BindOutput(const std::string& arg_name,
                            const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kOutput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }
    NewInstance& SetVersion(const std::string& version) {
      ParamTypeRegistry::Global().SetVersion(int_version(version),
                                             Split(kernel_type_, "/").front(),
                                             Place{target, precision, layout});
      return *this;
    }
    ///////////////////////////////////////////////////////////////////////
    // Funtion name: BindPaddleOpVersion
    // Author: DannyIsFunny
    // Description: Bind a kernel registry to a paddle op version, this
    //              fuction is not applicable on tiny_publish mode.
    ///////////////////////////////////////////////////////////////////////
    NewInstance& BindPaddleOpVersion(const std::string& op_type,
                                     int32_t version_id) {
#ifndef LITE_ON_TINY_PUBLISH
      ParamTypeRegistry::Global().BindPaddleOpVersion(
          op_type, version_id, kernel_type_, Place{target, precision, layout});
#endif
      return *this;
    }

    bool Finalize() { return true; }

   private:
    std::string kernel_type_;
  };

  template <IO io>
  void Register(const std::string& kernel_type,
                const Place& place,
                const std::string& arg_name,
                ParamType data_type) {
    KernelIdTy key{kernel_type, place, io, arg_name};
    types_[key] = data_type;
    CHECK(types_.count(key));
  }

  void SetVersion(const int64_t version,
                  const std::string& kernel_type,
                  const Place& place) {
    KernelIdTy key{kernel_type, place, IO(), std::string()};
    versions_[key] = version;
    CHECK(versions_.count(key));
  }

  int64_t GetVersion(const std::string& kernel_type, const Place& place) {
    KernelIdTy key{kernel_type, place, IO(), std::string()};
    if (versions_.count(key)) {
      return versions_[key];
    }
    return -1;
  }

#ifndef LITE_ON_TINY_PUBLISH
  ///////////////////////////////////////////////////////////////////////
  // Funtion name: BindPaddleOpVersion
  // Author: DannyIsFunny
  // Description: Bind a kernel registry to a paddle op version, this
  //              fuction is not applicable on tiny_publish mode.
  ///////////////////////////////////////////////////////////////////////
  void BindPaddleOpVersion(const std::string& op_type,
                           int32_t version,
                           const std::string& kernel_type,
                           const Place& place) {
    KernelIdTy key{kernel_type, place, IO(), std::string()};
    // Kernel registry can not bind to a op's vesion more than once.
    if (kernel_versions_.count(key) &&
        kernel_versions_[key].HasOpVersion(op_type)) {
      if (kernel_versions_[key].GetOpVersion(op_type) != version) {
        LOG(FATAL) << "Error: lite kernel (" << kernel_type
                   << ") has been bound to a paddle op (" << op_type
                   << ")'s version more than once, "
                   << "it's bound to version("
                   << kernel_versions_[key].GetOpVersion(op_type)
                   << ") before, but now rebound to another version ("
                   << version << ").";
      } else {
        return;
      }
    }
    // Bind current kernel to op(op_type) 's version.
    kernel_versions_[key].AddOpVersion(op_type, version);
    CHECK(kernel_versions_.count(key)) << "Error: failed to bind lite kernel ("
                                       << kernel_type << ") to op version of ("
                                       << op_type << ").";
  }

  ///////////////////////////////////////////////////////////////////////
  // Funtion name: GetKernelVersion
  // Author: DannyIsFunny
  // Description: Get kernel's version according to kernel type and place.
  ///////////////////////////////////////////////////////////////////////
  const KernelVersion& GetKernelVersion(const std::string& kernel_type,
                                        const Place& place) {
    KernelIdTy key{kernel_type, place, IO(), std::string()};
    return kernel_versions_[key];
  }
#endif

  const ParamType* RetrieveInArgument(const Place& place,
                                      const std::string& op_type,
                                      const std::string& arg_name) {
    return Retrieve<IO::kInput>(place, op_type, arg_name);
  }
  const ParamType* RetrieveOutArgument(const Place& place,
                                       const std::string& op_type,
                                       const std::string& arg_name) {
    return Retrieve<IO::kOutput>(place, op_type, arg_name);
  }

  static ParamTypeRegistry& Global() {
    static ParamTypeRegistry x;
    return x;
  }

  friend STL::ostream& operator<<(STL::ostream& os,
                                  const ParamTypeRegistry& other) {
    for (auto& item : other.types_) {
      os << item.first << " " << item.second.DebugString() << "\n";
    }
    return os;
  }

 protected:
  template <IO io>
  const ParamType* Retrieve(const Place& place,
                            const std::string& op_type,
                            const std::string& arg_name) {
    KernelIdTy key{op_type, place, io, arg_name};
    auto it = types_.find(key);
    if (it == types_.end()) return nullptr;
    return &it->second;
  }

 private:
  ParamTypeRegistry() = default;

 public:
  // Identification for a Kernel.
  struct KernelIdTy {
    std::string kernel_type;
    Place place;
    IO io;
    std::string arg_name;

    size_t hash() const;
    friend STL::ostream& operator<<(STL::ostream& os, const KernelIdTy& other);
  };

  using key_t = KernelIdTy;
  struct KeyCmp {
    bool operator()(const key_t& a, const key_t& b) const;
  };

 private:
  std::map<key_t, ParamType, ParamTypeRegistry::KeyCmp> types_;
  std::map<key_t, KernelVersion, ParamTypeRegistry::KeyCmp> kernel_versions_;
  std::map<key_t, int64_t, ParamTypeRegistry::KeyCmp> versions_;
};

}  // namespace lite
}  // namespace paddle
