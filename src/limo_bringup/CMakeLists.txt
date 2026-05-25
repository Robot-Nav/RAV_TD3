cmake_minimum_required(VERSION 3.0.2)

project(limo_bringup)

find_package(catkin REQUIRED COMPONENTS
  controller_manager
  joint_state_controller
  robot_state_publisher
  roscpp
  rospy
  std_msgs
  geometry_msgs
  move_base_msgs 
  actionlib
  tf
  cv_bridge 
  image_transport
  nav_msgs
)


catkin_package(

  # INCLUDE_DIRS include
  # LIBRARIES send_goal
  # CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
  # DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}

)

# add_executable(mbase src/goal.cpp)
# add_dependencies(mbase ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(mbase 
# ${catkin_LIBRARIES})