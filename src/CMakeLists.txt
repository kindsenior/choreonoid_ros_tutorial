choreonoid_add_simple_controller(RttTankController RttTankController.cpp)
target_link_libraries(RttTankController ${roscpp_LIBRARIES})

# inference
find_package(Torch REQUIRED)

# cmdvel
choreonoid_add_simple_controller(RtGo2InferenceCmdvelController RtGo2InferenceCmdvelController.cpp)
target_link_libraries(RtGo2InferenceCmdvelController ${roscpp_LIBRARIES} ${CHOREONOID_BASE_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET RtGo2InferenceCmdvelController PROPERTY CXX_STANDARD 17)

# joystick
choreonoid_add_simple_controller(RtGo2InferenceJoystickController RtGo2InferenceJoystickController.cpp)
target_link_libraries(RtGo2InferenceJoystickController ${roscpp_LIBRARIES} ${CHOREONOID_BASE_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET RtGo2InferenceJoystickController PROPERTY CXX_STANDARD 17)
