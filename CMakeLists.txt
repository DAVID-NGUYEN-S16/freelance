cmake_minimum_required(VERSION 3.26.4)
project(CV1 VERSION 0.1.0)

include(CTest)
enable_testing()

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
find_package (OpenCV 4.0.0 REQUIRED)
include_directories ("C:/Program Files/opencv/build")
add_executable(CV1 main.cpp)

target_link_libraries(CV1 ${OpenCV_LIBS})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)