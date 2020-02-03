# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# This code is used for getting current version, the lastest 
# tag is used as Paddle-lite version. If the lasted commit doesn't 
# match any existed tag, we will use the lasted commit id instead.

# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
set(TAG_VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")

while ("${PADDLE_VERSION}" STREQUAL "")
   # Get current tag corresponding to the lastest commit
   execute_process(
     COMMAND git describe --tags --exact-match
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
     OUTPUT_VARIABLE PADDLE_LITE_TAG
     RESULT_VARIABLE LITE_TAG_RESULT
     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
   )
   if (NOT ${LITE_TAG_RESULT})
      if(${PADDLE_LITE_TAG} MATCHES "v${TAG_VERSION_REGEX}")
         string(REPLACE "v" "" PADDLE_VERSION ${PADDLE_LITE_TAG})
      endif()
   endif()
   if ("${PADDLE_VERSION}" STREQUAL "")
      # If the lastest commit doesn't match any release tag, we get the lastest commit id in short-format instead.
      execute_process(
         COMMAND git log -1 --format=%h
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
         OUTPUT_VARIABLE PADDLE_LITE_COMMIT
         OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      set(PADDLE_VERSION ${PADDLE_LITE_COMMIT})
      message(WARNING "This project is not based on a release branch, we can't get a stable Lite version.")
   endif()
endwhile()

add_definitions(-DPADDLE_VERSION="${PADDLE_VERSION}")
message(STATUS "Paddle-Lite version is ${PADDLE_VERSION}")
