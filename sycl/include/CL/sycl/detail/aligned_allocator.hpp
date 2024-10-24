//==------------ aligned_allocator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <CL/sycl/detail/cnri.h>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/range.hpp>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {
template <typename T> class aligned_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

public:
  template <typename U> struct rebind { typedef aligned_allocator<U> other; };

  // Construct an object
  void construct(pointer Ptr, const_reference Val) {
    new (Ptr) value_type(Val);
  }

  // Destroy an object
  void destroy(pointer Ptr) { Ptr->~value_type(); }

  pointer address(reference Val) const { return &Val; }
  const_pointer address(const_reference Val) { return &Val; }

  // Allocate sufficiently aligned memory
  pointer allocate(size_t Size) {
    size_t NumBytes = Size * sizeof(value_type);
    const size_t Alignment =
        std::max<size_t>(getNextPowerOfTwo(sizeof(value_type)), 64);
    NumBytes = ((NumBytes - 1) | (Alignment - 1)) + 1;

    pointer Result = reinterpret_cast<pointer>(
        detail::OSUtil::alignedAlloc(Alignment, NumBytes));
    if (!Result)
      throw std::bad_alloc();
    return Result;
  }

  // Release allocated memory
  void deallocate(pointer Ptr, size_t size) {
    if (Ptr)
      detail::OSUtil::alignedFree(Ptr);
  }

  bool operator==(const aligned_allocator&) { return true; }
  bool operator!=(const aligned_allocator& rhs) { return false; }
};
} // namespace detail
} // namespace sycl
} // namespace cl
