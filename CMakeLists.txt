cmake_minimum_required(VERSION 3.5.0)
project(choreonoid_ros_tutorial)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  choreonoid
  )

catkin_package(SKIP_CMAKE_CONFIG_GENERATION SKIP_PKG_CONFIG_GENERATION)

include_directories(${catkin_INCLUDE_DIRS})

add_subdirectory(src)
