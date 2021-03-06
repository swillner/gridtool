# Copyright (C) 2019 Sven Willner <sven.willner@gmail.com>
#
# This file is part of gridtool.
#
# gridtool is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# gridtool is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with gridtool.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include(cmake/helpers.cmake)
include(cmake/libraries.cmake)

set_default_build_type(Release)

project(gridtool)

file(GLOB SOURCES src/*.cpp)
add_executable(gridtool ${SOURCES})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(gridtool PUBLIC -fno-math-errno -fno-rtti)
  add_custom_command(
    TARGET gridtool
    POST_BUILD
    COMMAND ${CMAKE_STRIP} gridtool
  )
endif()

target_include_directories(gridtool PRIVATE include lib/args lib/cpp-library lib/nvector)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel"
)# icc 19 does not yet fully support c++17, but we only need features it does support
  target_compile_options(gridtool PUBLIC -std=c++17)
else()
  target_compile_features(gridtool PUBLIC cxx_std_17)
endif()

add_system_paths()
set_advanced_options()
set_advanced_cpp_warnings(gridtool)
set_build_type_specifics(gridtool)
add_git_version(gridtool)

option(PARALLELIZATION "" ON)
if(PARALLELIZATION)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option(USE_CUDA "" OFF)
if(USE_CUDA)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  enable_language(CUDA)
  set_target_properties(gridtool PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set_source_files_properties(src/modules/downscaling.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(gridtool PRIVATE USE_CUDA)
endif()

option(WITH_GDAL "" OFF)
if(WITH_GDAL)
  # include GDAL library
  find_package(GDAL REQUIRED)
  message(STATUS "GDAL include directory: ${GDAL_INCLUDE_DIR}")
  message(STATUS "GDAL library: ${GDAL_LIBRARY}")
  mark_as_advanced(GDAL_CONFIG GDAL_INCLUDE_DIR GDAL_LIBRARY)
  target_link_libraries(gridtool PRIVATE gdal)
  target_include_directories(gridtool PRIVATE ${GDAL_INCLUDE_DIR})
  target_compile_definitions(gridtool PRIVATE GRIDTOOL_WITH_GDAL)
endif()

include_netcdf_cxx4(gridtool ON v4.3.0)

find_path(CIMG_INCLUDE_DIR NAMES "CImg.h")
mark_as_advanced(CIMG_INCLUDE_DIR)
if (CIMG_INCLUDE_DIR)
  message(STATUS "CImg include directory: ${CIMG_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "CImg not found")
endif()
target_include_directories(gridtool PRIVATE ${CIMG_INCLUDE_DIR})

find_package(X11 REQUIRED)
target_include_directories(gridtool PRIVATE ${X11_INCLUDE_DIR})
target_link_libraries(gridtool PRIVATE  ${X11_LIBRARIES})

add_cpp_tools(gridtool)
