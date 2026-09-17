cmake_minimum_required(VERSION 3.28)

# Architecture is fixed before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target. The implementation is tuned on
# sm_120a (RTX 5090) and also builds for sm_89 (Ada, e.g. RTX 4060); non-Blackwell targets
# omit the Blackwell-only NVFP4/TMA kernels selected by NINFER_ENABLE_NVFP4 below.
set(NINFER_SUPPORTED_CUDA_ARCHITECTURES 89 120a)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
foreach(_ninfer_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(NOT _ninfer_arch IN_LIST NINFER_SUPPORTED_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "NInfer supports CMAKE_CUDA_ARCHITECTURES from "
      "${NINFER_SUPPORTED_CUDA_ARCHITECTURES}; got '${_ninfer_arch}'. Extend the list only "
      "after validating the target: the kernels below are qualified against sm_89 and sm_120a.")
  endif()
endforeach()

# NVFP4 (w4a4) kernels and their TMA / fp4-value KV-cache companions need ptxas features that
# first appear in sm_100a (notably .kind::mxf4nvf4 and cvt.e2m1x2). Other targets exclude them
# and their host entry points fail loudly instead of silently degrading.
set(NINFER_ENABLE_NVFP4 OFF)
foreach(_ninfer_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_ninfer_arch MATCHES "^(100|100a|120|120a)$")
    set(NINFER_ENABLE_NVFP4 ON)
  endif()
endforeach()
unset(_ninfer_arch)

# Programmatic dependent launch lowers to griddepcontrol.{launch_dependents,wait}, which first
# appear in sm_90. Older targets keep the same source but launch with ordinary stream ordering
# and compile the device hooks to nothing; only the producer/consumer overlap is lost.
set(NINFER_ENABLE_PDL OFF)
foreach(_ninfer_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_ninfer_arch MATCHES "^(90|90a|100|100a|120|120a)$")
    set(NINFER_ENABLE_PDL ON)
  endif()
endforeach()
unset(_ninfer_arch)

# Large static __shared__ arrays. ptxas accepts a static shared array larger than the 48 KiB
# per-block limit only from sm_100a on. Measured with the CUDA 13.2 ptxas on this repository's
# kernels: sm_89 and sm_90 cap static shared at 0xc000, sm_100a at 0x38c00 and sm_120a at 0x18c00.
# Seven q8 kernel families stage their K-split tiles above 0xc000, so those targets exclude them
# and reject the routes they served instead of emitting code ptxas cannot assemble.
set(NINFER_ENABLE_LARGE_STATIC_SMEM OFF)
foreach(_ninfer_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_ninfer_arch MATCHES "^(100|100a|120|120a)$")
    set(NINFER_ENABLE_LARGE_STATIC_SMEM ON)
  endif()
endforeach()
unset(_ninfer_arch)

message(STATUS "NInfer CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} "
               "(NVFP4 kernels: ${NINFER_ENABLE_NVFP4}; PDL: ${NINFER_ENABLE_PDL}; "
               "large static __shared__: ${NINFER_ENABLE_LARGE_STATIC_SMEM})")

project(ninfer LANGUAGES C CXX CUDA)

# Single project-wide authority for the hardware capability macros. Every translation unit that
# includes a guarded header (ops entry points, paged KV storage geometry, serve option parsing,
# programmatic launch helpers) must agree, so these are global definitions rather than per-target
# ones.
if(NINFER_ENABLE_NVFP4)
  add_compile_definitions(NINFER_ENABLE_NVFP4=1)
else()
  add_compile_definitions(NINFER_ENABLE_NVFP4=0)
endif()
if(NINFER_ENABLE_PDL)
  add_compile_definitions(NINFER_ENABLE_PDL=1)
else()
  add_compile_definitions(NINFER_ENABLE_PDL=0)
endif()
if(NINFER_ENABLE_LARGE_STATIC_SMEM)
  add_compile_definitions(NINFER_ENABLE_LARGE_STATIC_SMEM=1)
else()
  add_compile_definitions(NINFER_ENABLE_LARGE_STATIC_SMEM=0)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# A native MSVC build needs three settings that GCC/Clang on Linux and WSL supply by default:
#
#   /Zc:preprocessor  CCCL refuses the traditional MSVC preprocessor, and says so itself:
#                       cuda/std/__cccl/preprocessor.h: fatal error C1189: MSVC/cl.exe with
#                       traditional preprocessor is used. ... passing `/Zc:preprocessor` to cl.exe
#                     This is reached by any CUDA source that uses <cuda/std/...>; the check lives in
#                     a CCCL header that <cub/...> alone does not pull in, so a small probe does not
#                     reproduce it.
#   /utf-8            spdlog's bundled {fmt} refuses to build without it, failing every spdlog
#                     translation unit with
#                       static_assert failed: 'Unicode support requires compiling with /utf-8'
#                     Every source in this repository is UTF-8, so the flag matches the encoding the
#                     sources are actually stored in.
#   NOMINMAX          <windows.h> defines `min` and `max` as function-like macros, which breaks this
#                     project's own `std::min` / `std::max` / `std::numeric_limits<T>::max()` uses;
#                     each call is reported as `error C4003: not enough arguments for function-like
#                     macro invocation of 'max'` followed by a cascade of syntax errors. NVTX's
#                     nvtxDetail/nvtxImpl.h includes <windows.h> without this guard, and
#                     src/core/nvtx.h is reached from src/core/weight.h, so the pollution reached
#                     every translation unit that touches an Op or a model. The guard belongs at
#                     project scope rather than in one header: two independent dependencies here
#                     already pull in <windows.h>, and only spdlog guards it itself.
#
# The two compiler switches must be scoped by language. CMake propagates host flags into the nvcc
# command line, and a bare `/Zc:preprocessor` or `/utf-8` there is taken as an input file rather than
# an option, which fails every CUDA translation unit with
#   nvcc fatal : A single input file is required for a non-link phase when an outputfile is specified
# CUDA sources therefore receive the same settings through the host-compiler passthrough. A macro
# definition needs no such scoping, and CUDA sources include src/core/nvtx.h too.
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  add_compile_definitions(NOMINMAX)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
