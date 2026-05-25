cmake_minimum_required(VERSION 3.0.2)
project(limoRL)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_srvs
  std_msgs
  tf
)


catkin_package(
  CATKIN_DEPENDS gazebo_msgs geometry_msgs nav_msgs roscpp rospy sensor_msgs std_srvs std_msgs tf
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



catkin_install_python(PROGRAMS
  scripts/limo_td3.py
  scripts/limo_td3_test.py
  scripts/unit_1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
