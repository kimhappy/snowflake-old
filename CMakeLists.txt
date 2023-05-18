cmake_minimum_required ( VERSION 3.26 )

project ( snowflake )

file ( GLOB_RECURSE SNOWFLAKE_HEADERS
  "include/snowflake/*.hpp"
)

add_library ( ${PROJECT_NAME} INTERFACE ${SNOWFLAKE_HEADERS} )

target_include_directories ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )
