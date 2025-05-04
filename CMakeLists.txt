cmake_minimum_required(VERSION 3.0.0)

project(yolov12infer)
set(PROJECT_NAME yolov12infer)
set(ONNXRUNTIME_DIR /Users/macbook/onnxruntime)
# option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)
find_package(ONNX REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")

# Add executable for image inference
add_executable(${PROJECT_NAME} src/main.cpp src/yolov12.cpp src/utils.cpp src/logger.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set include directories for all executables
target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Set compile features for all executables
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
# Link libraries for all executables
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_MODE)

message(STATUS "We are building on macOS!")
# Specific macOS build commands or flags
target_link_libraries(${PROJECT_NAME} onnxruntime)
