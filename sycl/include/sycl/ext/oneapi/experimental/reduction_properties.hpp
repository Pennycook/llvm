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

template <typename BinaryOperation, typename PropertyList>
auto WrapOp(BinaryOperation combiner, PropertyList properties) {
  if constexpr (properties.template has_property<deterministic_key>()) {
    return DeterministicOperatorWrapper(combiner);
  }
  else {
    return combiner;
  }
}

template <typename PropertyList>
property_list GetReductionPropertyList(PropertyList properties) {
  if constexpr (properties.template has_property<initialize_to_identity_key>()) {
    return sycl::property::reduction::initialize_to_identity{};
  }
  return {};
}

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {

template <typename BinaryOperation>
struct DeterministicOperatorWrapper {

  DeterministicOperatorWrapper(BinaryOperation BinOp = BinaryOperation()) : BinOp(BinOp) {}

  template <typename... Args>
  std::invoke_result_t<BinaryOperation, Args...>
  operator()(Args... args) {
    return BinOp(std::forward<Args>(args)...);
  }

  BinaryOperation BinOp;

};

template <typename BinaryOperation>
struct IsDeterministicOperator<DeterministicOperatorWrapper<BinaryOperation>> : std::true_type {};

} // namespace detail

template <typename BufferT, typename BinaryOperation, typename PropertyList>
auto reduction(BufferT vars, handler& cgh, BinaryOperation combiner,
               PropertyList properties) {
  auto WrappedOp = ext::oneapi::experimental::detail::WrapOp(combiner, properties);
  auto RuntimeProps = ext::oneapi::experimental::detail::GetReductionPropertyList(properties);
  return reduction(vars, cgh, WrappedOp, RuntimeProps);
}

#if 0
template <typename T, typename BinaryOperation, typename PropertyList>
__unspecified__ reduction(T* var, BinaryOperation combiner,
                          PropertyList properties);
template <typename T, typename Extent, typename BinaryOperation, typename PropertyList>
__unspecified__ reduction(span<T, Extent> vars, BinaryOperation combiner,
                          PropertyList properties);
template <typename BufferT, typename BinaryOperation, typename PropertyList>
__unspecified__
reduction(BufferT vars, handler& cgh, const BufferT::value_type& identity,
          BinaryOperation combiner, PropertyList properties);
template <typename T, typename BinaryOperation, typename PropertyList>
__unspecified__ reduction(T* var, const T& identity, BinaryOperation combiner,
                          PropertyList properties);
template <typename T, typename Extent, typename BinaryOperation, typename PropertyList>
__unspecified__ reduction(span<T, Extent> vars, const T& identity,
                          BinaryOperation combiner,
                          PropertyList properties);
#endif

} // namespace _V1
} // namespace sycl
