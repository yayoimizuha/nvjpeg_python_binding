cmake_minimum_required(VERSION 3.28)
project(detect_face CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++20")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++2a")

find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)

#find_package(CUDA REQUIRED)
#find_package(NVJPEG REQUIRED)
message("CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message("CUDA Toolkit Library dir: ${CUDAToolkit_LIBRARY_DIR}")
message("CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
set(nvjpeg_library ${CUDAToolkit_LIBRARY_DIR}/nvjpeg.lib)

set(pybind11_DIR [=[C:\Users\tomokazu\miniconda3\envs\helloproject-ai\Library\share\cmake\pybind11]=])
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_executable(detect_face main.cu)
add_executable(test test.cu)
set_target_properties(detect_face PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(detect_face ${nvjpeg_library})
add_library(pythonlib MODULE pybind11.cu)
set_target_properties(pythonlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(pythonlib PRIVATE CUDA::nvjpeg pybind11::module pybind11::lto pybind11::windows_extras Thrust::CUDA CUDA::nppig)
target_include_directories(pythonlib PUBLIC "${pybind11_INCLUDE_DIR}")
pybind11_extension(pythonlib)

set(onnxruntime_gpu_dir [=[C:/Program Files/onnxruntime/]=])
target_include_directories(detect_face PUBLIC "${onnxruntime_gpu_dir}/include")
target_link_directories(detect_face PUBLIC "${onnxruntime_gpu_dir}/lib")
#target_include_directories(detect_face PUBLIC [=[C:\Users\tomokazu\.nuget\packages\microsoft.ml.onnxruntime.gpu\1.16.1\build\native\include]=])
#target_link_directories(detect_face PUBLIC [=[C:\Users\tomokazu\.nuget\packages\microsoft.ml.onnxruntime.gpu\1.16.1\runtimes\win-x64\native]=])

target_link_libraries(detect_face onnxruntime)
target_link_libraries(detect_face onnxruntime_providers_tensorrt)
thrust_create_target(Thrust)
target_link_libraries(detect_face Thrust)
#target_link_libraries(pythonlib Thrust)

target_link_libraries(test CUDA::nvjpeg pybind11::module pybind11::lto pybind11::windows_extras Thrust::CUDA CUDA::nppig)
