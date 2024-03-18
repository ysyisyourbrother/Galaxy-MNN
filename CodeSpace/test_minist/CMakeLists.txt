cmake_minimum_required(VERSION 3.10)
project(mnist)
set (EXTRA_INCLUDE "")
set (EXTRA_LIBS "")
set (EXTRA_OPTIONS "")

if(MNN_CUDA_PROFILE)
    set(CUDA_MIN_VERSION "9.0")
    find_package(CUDA ${CUDA_MIN_VERSION})
    MESSAGE("benchmark message:CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
    link_directories(${CUDA_INCLUDE_DIRS}/../lib/
        ${CUDA_INCLUDE_DIRS}/../lib64/)
    set(EXTRA_INCLUDE ${CUDA_INCLUDE_DIRS})
    set(EXTRA_LIBS  -lnvToolsExt)
    set(EXTRA_OPTIONS -DMNN_CUDA_PROFILE)
endif()

set(INFERENCE_SRC ${CMAKE_CURRENT_LIST_DIR}/inference.cpp)

set(CMAKE_CXX_STANDARD 11)
add_executable(mnist.out ${INFERENCE_SRC})
target_include_directories(mnist.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/ ${CMAKE_CURRENT_SOURCE_DIR}/tools/ ${EXTRA_INCLUDE})
target_compile_options(mnist.out PRIVATE ${EXTRA_OPTIONS})
target_link_libraries(mnist.out ${MNN_DEPS} ${EXTRA_LIBS})




