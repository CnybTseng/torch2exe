cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(algorithm)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	if (${MSVC_TOOLSET_VERSION} EQUAL 141)			# Visual Studio 2017
		set(VC vc15)
	elseif (${MSVC_TOOLSET_VERSION} EQUAL 142)		# Visual Studio 2019
		set(VC vc16)
	endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.3/x64/${VC}/staticlib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.3/lib/cmake/opencv4)
endif()

cmake_policy(SET CMP0054 NEW)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_package(Threads)
	find_package(Iconv)
endif()

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../install/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.3/include)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../install/lib)
link_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.3/x64/${VC}/staticlib)

add_definitions(-DALGORITHM_DLL)

set(CMAKE_CXX_STANDARD 11)

if(MSVC)
	set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
    )
	foreach(CompilerFlag ${CompilerFlags})
		string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
	endforeach()
endif()

add_executable(test-${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(test-${PROJECT_NAME} ${PROJECT_NAME} opencv_core opencv_imgproc opencv_imgcodecs)