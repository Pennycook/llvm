//===-- cnri.h - SYCL common native runtime interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This source is the definition of the SYCL Common Native Runtime Interface
/// (CNRI). It is the interface between the device-agnostic SYCL runtime layer
/// and underlying "native" runtimes such as OpenCL.

#pragma once

#include "CL/opencl.h"

#include <stdint.h>

/// Target identification strings
#define CNRI_TGT_STR_UNKNOWN "<unknown>"
#define CNRI_TGT_STR_SPIRV32 "spir"
#define CNRI_TGT_STR_SPIRV64 "spir64"

/// Kinds of device images
enum cnri_device_image_format {
  CNRI_IMG_NONE,   // image format is not determined
  CNRI_IMG_NATIVE, // image format is specific to a device
  // portable image kinds go next
  CNRI_IMG_SPIRV,         // SPIR-V
  CNRI_IMG_LLVMIR_BITCODE // LLVM bitcode
};

typedef void __tgt_offload_entry;

// Device image descriptor version supported by this library.
#define CNRI_DEVICE_IMAGE_STRUCT_VERSION ((uint16_t)1)
#define SYCL_OFFLOAD_KIND ((uint8_t)4)

/// This struct is a record of the device image information. If the Kind field
/// denotes a portable image kind (SPIRV or LLVMIR), the DeviceTargetSpec field
/// can still be specific and denote e.g. FPGA target.
/// It must match the __tgt_device_image structure generated by
/// the clang-offload-wrapper tool when their Version field match.
struct cnri_device_image {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// the kind of offload model the image employs; must be 4 for SYCL
  uint8_t Kind;
  /// format of the image data - SPIRV, LLVMIR bitcode,...
  uint8_t Format;
  /// null-terminated string representation of the device's target architecture
  const char *DeviceTargetSpec;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "build" program at runtime
  const char *BuildOptions;
  /// Pointer to the manifest data start
  const unsigned char *ManifestStart;
  /// Pointer to the manifest data end
  const unsigned char *ManifestEnd;
  /// Pointer to the target code start
  const unsigned char *ImageStart;
  /// Pointer to the target code end
  const unsigned char *ImageEnd;
  /// the offload entry table (not used, for compatibility with OpenMP)
  __tgt_offload_entry *EntriesBegin;
  __tgt_offload_entry *EntriesEnd;
};

// Offload binary descriptor version supported by this library.
#define CNRI_BIN_DESC_STRUCT_VERSION ((uint16_t)1)

/// This struct is a record of all the device code that may be offloaded.
/// It must match the __tgt_bin_desc structure generated by
/// the clang-offload-wrapper tool when their Version field match.
struct cnri_bin_desc {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// Number of device binary images in this descriptor
  uint16_t NumDeviceImages;
  /// Device binary images data
  cnri_device_image *DeviceImages;
  /// the offload entry table (not used, for compatibility with OpenMP)
  __tgt_offload_entry *HostEntriesBegin;
  __tgt_offload_entry *HostEntriesEnd;
};

// TODO For now code below is a placeholder for future real implementation
typedef cl_context cnri_context;
typedef cl_event cnri_event;
typedef cl_program cnri_program;
typedef cl_kernel cnri_kernel;

enum { CNRI_SUCCESS = CL_SUCCESS };

// redirections to OpenCL
#define cnriReleaseProgram clReleaseProgram
#define cnriRetainProgram clRetainProgram

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// CNRI unique APIs

/// Selects the most appropriate device image based on runtime information and
/// the image characteristics
cl_int cnriSelectDeviceImage(cnri_context ctx, cnri_device_image **images,
                             cl_uint num_images,
                             cnri_device_image **selected_image);

#ifdef __cplusplus
}
#endif // __cplusplus

#define CHECK_CNRI_CODE(x) CHECK_OCL_CODE(x)
#define CHECK_CNRI_CODE_NO_EXC(x) CHECK_OCL_CODE_NO_EXC(x)
