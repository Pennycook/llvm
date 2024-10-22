//==------- properties.hpp - SYCL properties associated with reductions ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#define SYCL_EXT_ONEAPI_REDUCTION_PROPERTIES

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

struct deterministic_key {
  using value_t = property_value<deterministic_key>;
};
inline constexpr deterministic_key::value_t deterministic;

struct initialize_to_identity_key {
  using value_t = property_value<initialize_to_identity_key>;
};
inline constexpr initialize_to_identity_key::value_t initialize_to_identity;

template <> struct is_property_key<deterministic_key> : std::true_type {};
template <> struct is_property_key<initialize_to_identity_key> : std::true_type {};

namespace detail {

template <>
struct IsCompileTimeProperty<deterministic_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<initialize_to_identity_key> : std::true_type {};

template <> struct PropertyToKind<deterministic_key> {
  static constexpr PropKind Kind = PropKind::Deterministic;
};
template <> struct PropertyToKind<initialize_to_identity_key> {
  static constexpr PropKind Kind = PropKind::InitializeToIdentity;
};

template <typename BinaryOperation>
struct DeterministicOperatorWrapper {

  DeterministicOperatorWrapper(BinaryOperation BinOp) : BinOp(BinOp) {}

  template <typename... Args>
  std::invoke_result_t<BinaryOperation, Args...>
  operator()(Args... args) {
    return BinOp(std::forward<Args>(args)...);
  }

  BinaryOperation& BinOp;

};

#ifdef SYCL_DETERMINISTIC_REDUCTION
// Act as if all operators require determinism.
template <typename T> struct IsDeterministicOperator : std::true_type {};
#else
// Each operator declares whether determinism is required.
template <typename T> struct IsDeterministicOperator : std::false_type {};

template <typename BinaryOperation>
struct IsDeterministicOperator<DeterministicOperatorWrapper<BinaryOperation>> : std::true_type {};
#endif

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
