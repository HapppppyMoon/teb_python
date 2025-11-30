cmake_minimum_required(VERSION 3.14)
project(teb_local_planner VERSION 1.0.0)

# Release 빌드
set(CMAKE_BUILD_TYPE Release)

# C++17 (required by g2o)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python 바인딩 옵션
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# 의존성 찾기
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

set(EXTERNAL_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${Boost_LIBRARIES} ${SUITESPARSE_LIBRARIES} ${G2O_LIBRARIES})

# 라이브러리 빌드
add_library(teb_local_planner SHARED
   src/timed_elastic_band.cpp
   src/optimal_planner.cpp
   src/obstacles.cpp
   src/teb_config.cpp
   src/homotopy_class_planner.cpp
   src/graph_search.cpp
   src/costmap_converter.cpp
)

target_include_directories(teb_local_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
  ${EXTERNAL_INCLUDE_DIRS}
)

target_link_libraries(teb_local_planner PUBLIC
   ${EXTERNAL_LIBS}
)

# Python 바인딩 (선택)
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(pyteb python/bindings.cpp)
    target_link_libraries(pyteb PRIVATE teb_local_planner)
  else()
    message(WARNING "pybind11 not found, Python bindings will not be built")
  endif()
endif()

# 설치
install(TARGETS teb_local_planner
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
   DESTINATION include
)
