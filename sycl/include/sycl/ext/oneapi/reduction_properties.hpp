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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
