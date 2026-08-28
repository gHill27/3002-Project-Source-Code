cmake_minimum_required(VERSION 3.8)
project(exploration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# --- C++ novelty mapping node ---
add_executable(novelty_mapping_node
  src/novelty_mapping.cpp
)

target_include_directories(novelty_mapping_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(novelty_mapping_node PUBLIC cxx_std_17)

ament_target_dependencies(novelty_mapping_node
  rclcpp
  nav_msgs
)

install(TARGETS novelty_mapping_node
  DESTINATION lib/${PROJECT_NAME}
)
# --------------------------------

install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  # TODO: add more nodes here
  #exploration/novelty_mapping.py
  exploration/localizer.py
  exploration/novelty_mapping.py
  DESTINATION lib/exploration
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()