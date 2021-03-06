cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
project( atracker )

find_package(OpenCV REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(EXECUTABLE_OUTPUT_PATH ../bin)

include_directories( src )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:/ThirdLib/PCL 1.8.0/3rdParty/Boost/include/boost-1_59")

file(GLOB_RECURSE atracker_SRC "src/*.cpp" "src/*.h")
add_executable( atracker ${atracker_SRC})
target_link_libraries( atracker ${OpenCV_LIBS} )
