set(LIGHT_API_SRC light_api.cc light_api_impl.cc paddle_place.cc paddle_api.cc)
set(FILL_API_SRC cxx_api.cc cxx_api_impl.cc)


add_library(api OBJECT ${LIGHT_API_SRC})
add_library(full_api OBJECT ${FILL_API_SRC})
