if (NOT LITE_WITH_PYTHON)
    return()
endif()

# to create setup.py for packeting whl for Paddle Lite and opt

execute_process(
  COMMAND git describe --tags --exact-match
  WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LITE_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LITE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(APPLE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup_mac.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
endif()
