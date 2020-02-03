message(STATUS "GetInto version.cmake")
# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
set(tmp_version "HEAD")
set(TAG_VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
set(COMMIT_VERSION_REGEX "[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+")
set(LATEST_PADDLE_VERSION "0.0.0")

while ("${PADDLE_VERSION}" STREQUAL "")
  # Check current branch name
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref ${tmp_version}
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH_NAME
    RESULT_VARIABLE GIT_BRANCH_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "GIT_BRANCH_NAME:${GIT_BRANCH_NAME}")
  if (NOT ${GIT_BRANCH_RESULT})
    # get the lasted commit id
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list --tags --max-count=1
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_TAG_NUM
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Lasted commit ID: ${GIT_TAG_NUM}")
    # Get the latested tag corresponding to the lasted commit
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags ${GIT_TAG_NUM}
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_TAG_NAME
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (NOT ${GIT_RESULT} AND ${GIT_TAG_NAME} MATCHES "v${TAG_VERSION_REGEX}")
      string(REPLACE "v" "" PADDLE_VERSION ${GIT_TAG_NAME})
    else()
      set(PADDLE_VERSION "0.0.0")
      message(WARNING "Cannot add paddle version from git tag")
    endif()
  endif()
endwhile()

add_definitions(-DPADDLE_VERSION=${PADDLE_VERSION})
message(STATUS "Paddle version is ${PADDLE_VERSION}")
