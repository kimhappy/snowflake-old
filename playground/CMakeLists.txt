cmake_minimum_required ( VERSION 3.12 )

project ( playground )

add_subdirectory ( ../ snowflake )

set ( CMAKE_BUILD_TYPE              Release )
set ( CMAKE_CXX_STANDARD            20      )
set ( CMAKE_EXPORT_COMPILE_COMMANDS ON      )

set ( CMAKE_CXX_FLAGS_RELEASE "-Ofast -Wall -Wextra -fno-exceptions -fvisibility=hidden -fvisibility-inlines-hidden -march=native" )

file ( GLOB GWAVIO_SOURCES
  thirdparty/gwavio/src/internal/*.cpp
)

file ( GLOB PLAYGROUND_SOURCES
  src/*.cpp
)

add_library ( gwavio
  ${GWAVIO_SOURCES}
)

add_executable ( ${PROJECT_NAME}
  ${PLAYGROUND_SOURCES}
)

target_include_directories( gwavio PUBLIC
  thirdparty/gwavio/src/include
)

target_include_directories ( ${PROJECT_NAME} PUBLIC
  include/
)

target_link_libraries ( ${PROJECT_NAME} snowflake gwavio )

set_property ( TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE )
