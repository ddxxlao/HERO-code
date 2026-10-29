# tools/vision - hero-vision, the image encoder for HERO's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DHERO_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DHERO_PORTABLE=OFF]
#         [-DHERO_VISION_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx (oneAPI sourced; Linux)]
#         [-DHERO_VISION_HIP=ON -DHERO_VISION_HIP_ARCHS="gfx1200;gfx1201" + the HIP compiler the engine build uses]
#   cmake --build build-vision --target hero-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DHERO_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(hero_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(HERO_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(HERO_VISION_HIP "run the vision encoder on an AMD GPU (builds ggml-hip); the program is named hero-vision-hip" OFF)
option(HERO_VISION_SYCL "run the vision encoder on an Intel GPU (builds ggml-sycl, needs the oneAPI icx/icpx compilers); the program is named hero-vision-sycl" OFF)
set(HERO_VISION_HIP_ARCHS "" CACHE STRING "the gfx architectures hero-vision-hip is compiled for (';'-separated)")
if(HERO_VISION_HIP AND HERO_VISION_CUDA)
  message(FATAL_ERROR "hero-vision: HERO_VISION_CUDA and HERO_VISION_HIP are two different programs")
endif()
if(HERO_VISION_SYCL AND (HERO_VISION_CUDA OR HERO_VISION_HIP))
  message(FATAL_ERROR "hero-vision: HERO_VISION_SYCL is its own program: not together with CUDA or HIP")
endif()
if(HERO_VISION_HIP AND NOT HERO_VISION_HIP_ARCHS)
  message(FATAL_ERROR "hero-vision: HERO_VISION_HIP needs HERO_VISION_HIP_ARCHS, for example \"gfx1200;gfx1201\"")
endif()
option(HERO_VISION_METAL "a Mac: ggml's Metal backend in the program, so --gpu reads pictures on the Mac's GPU (OFF: CPU only)" ON)
option(HERO_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
if(HERO_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(HERO_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "hero-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "hero-vision: native build for this CPU (HERO_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${HERO_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_HIP ${HERO_VISION_HIP} CACHE BOOL "" FORCE)
if(APPLE)
  # One program for both readers: hero-vision --gpu runs the encoder on the Metal GPU, without --gpu on the CPU (Apple Silicon always
  # has the Metal framework, the engine needs it too).  The Metal shaders are embedded in the binary and compiled by macOS when the
  # reader starts, so nothing but the program has to be shipped.  No -march flag: the default target (an M1) runs on every Apple
  # Silicon Mac, and ggml's CPU path has no BF16 vector kernel on ARM (measured on an M4: the same ~50 s for a 260-token picture with
  # -march=armv8.6-a+bf16+i8mm as without), so a faster target would only risk the M1.
  set(GGML_METAL ${HERO_VISION_METAL} CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ${HERO_VISION_METAL} CACHE BOOL "" FORCE)
endif()
set(GGML_SYCL ${HERO_VISION_SYCL} CACHE BOOL "" FORCE)
if(HERO_VISION_SYCL)
  set(GGML_SYCL_TARGET "INTEL" CACHE STRING "" FORCE)
endif()
if(HERO_VISION_HIP)
  # the engine's HIP build says the same (cmake/hip_backend.cmake); ggml-hip reads AMDGPU_TARGETS / CMAKE_HIP_ARCHITECTURES
  set(AMDGPU_TARGETS "${HERO_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
  set(GPU_TARGETS "${HERO_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
  set(CMAKE_HIP_ARCHITECTURES "${HERO_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
  # #3359: a Linux toolchain links a PIE; ggml-hip's objects must be position independent (the engine's HIP build says the same in
  # cmake/hip_backend.cmake).  Only for the HIP reader: the other readers' builds stay as they were.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(hero-vision hero_vision.cpp)
if(MSVC)   # #3202: the ANSI code page of the process is UTF-8, so a Cyrillic model folder reaches the reader intact (see ../../CMakeLists.txt)
  target_sources(hero-vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hero-utf8.manifest)
endif()
target_link_libraries(hero-vision PRIVATE mtmd llama)
target_compile_features(hero-vision PRIVATE cxx_std_17)
set_target_properties(hero-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(HERO_VISION_HIP)
  # beside hero (and, in a component, the CPU reader "hero-vision"): the GPU reader is its own program, so a PC where
  # the HIP runtime does not load still has the CPU one to fall back on
  set_target_properties(hero-vision PROPERTIES OUTPUT_NAME hero-vision-hip)
  string(REPLACE ";" "," _hip_archs "${HERO_VISION_HIP_ARCHS}")
  target_compile_definitions(hero-vision PRIVATE HERO_VISION_HIP HERO_VISION_HIP_ARCHS="${_hip_archs}")
endif()
if(APPLE AND HERO_VISION_METAL)
  target_compile_definitions(hero-vision PRIVATE HERO_VISION_METAL)
endif()
if(HERO_VISION_SYCL)
  # the Intel GPU reader is its own program too, beside the CPU reader "hero-vision" it falls back on
  set_target_properties(hero-vision PROPERTIES OUTPUT_NAME hero-vision-sycl)
  target_compile_definitions(hero-vision PRIVATE HERO_VISION_SYCL)
endif()
if(WIN32 AND HERO_VISION_HIP)
  # hipblas.dll (and rocblas.dll behind it) live in rocm\bin of the engine package: loaded on first use, after main() has put that folder on
  # the DLL search path, so the reader starts from any PATH and says "no GPU" itself where there is none
  target_link_libraries(hero-vision PRIVATE delayimp)
  target_link_options(hero-vision PRIVATE -Wl,/delayload:hipblas.dll)
endif()
if(UNIX AND NOT APPLE)
  # the libraries it needs are the ones in its own folder (a component folder is relocatable): RUNPATH $ORIGIN
  set_target_properties(hero-vision PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
  target_link_options(hero-vision PRIVATE -Wl,--enable-new-dtags)
endif()
if(HERO_VISION_SYCL AND UNIX AND NOT APPLE)
  # libsycl and the OpenMP runtime live in the oneAPI compiler's lib folder, which only a sourced setvars.sh puts on the search path:
  # recorded in the program, it starts from a plain shell (the server starts it with the PC's own environment)
  get_filename_component(_icpx_bin "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_icpx_bin "${_icpx_bin}" DIRECTORY)
  get_filename_component(_icpx_lib "${_icpx_bin}/../lib" ABSOLUTE)
  set_target_properties(hero-vision PROPERTIES INSTALL_RPATH "$ORIGIN;${_icpx_lib}")
endif()

# opens a GGUF that has tensor types llama.cpp does not know (a Gyro quantization's TQK6 / TQK7) for its vocabulary only
add_executable(hero-vision-gguf-test EXCLUDE_FROM_ALL gguf_meta_test.cpp)
target_link_libraries(hero-vision-gguf-test PRIVATE llama)
target_compile_features(hero-vision-gguf-test PRIVATE cxx_std_17)
set_target_properties(hero-vision-gguf-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# the HIP reader's architecture check (gpu_arch.h), without a GPU
add_executable(hero-vision-arch-test EXCLUDE_FROM_ALL gpu_arch_test.cpp)
target_compile_features(hero-vision-arch-test PRIVATE cxx_std_17)
set_target_properties(hero-vision-arch-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
