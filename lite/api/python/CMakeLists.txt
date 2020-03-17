if (NOT LITE_WITH_PYTHON)
    return()
endif()

# to create setup.py for packeting whl for Paddle-Lite and opt
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_subdirectory(pybind)
#add_subdirectory(interface)
