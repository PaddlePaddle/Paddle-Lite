if (((NOT LITE_WITH_ARM) AND (NOT LITE_WITH_X86)) OR (NOT LITE_WITH_JAVA))
  return()
endif()

add_subdirectory(jni)
