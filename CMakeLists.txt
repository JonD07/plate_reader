cmake_minimum_required(VERSION 3.5)
project(plate_reader)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# --- the service definition lives in this package ----------------------
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ReadPlate.srv"
)

# --- OpenCV and Tesseract (Ubuntu 20.04 / Foxy system packages) --------
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED IMPORTED_TARGET tesseract)
pkg_check_modules(LEPTONICA REQUIRED IMPORTED_TARGET lept)

add_library(alpr STATIC src/alpr.cpp)
target_include_directories(alpr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(alpr PUBLIC
  ${OpenCV_LIBS}
  PkgConfig::TESSERACT
  PkgConfig::LEPTONICA
)

add_executable(plate_reader_node src/plate_reader_node.cpp)
target_include_directories(plate_reader_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(plate_reader_node alpr)
ament_target_dependencies(plate_reader_node rclcpp)

# link the node against this package's generated service typesupport
# (Foxy-era macro; replaced by rosidl_get_typesupport_target in later
# distros — see README for Jazzy notes)
rosidl_target_interfaces(plate_reader_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS plate_reader_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
