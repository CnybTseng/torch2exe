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
	set(WHICH where)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(WHICH which)
else()
	message(FATAL_ERROR "Unknown system")
	return()
endif()

execute_process(COMMAND ${WHICH} nvcc OUTPUT_VARIABLE NVCC_FULL_PATH)
cmake_path(CONVERT ${NVCC_FULL_PATH} TO_CMAKE_PATH_LIST NVCC_FULL_PATH_CMAKE_STYLE)
cmake_path(GET NVCC_FULL_PATH_CMAKE_STYLE PARENT_PATH NVCC_PATH)
cmake_path(GET NVCC_PATH PARENT_PATH NVCC_ROOT_PATH)
set(CUDA_INCLUDE_PATH "${NVCC_ROOT_PATH}/include")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(CUDA_LIBRARY_PATH "${NVCC_ROOT_PATH}/lib/x64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(CUDA_LIBRARY_PATH "${NVCC_ROOT_PATH}/lib64")
endif()

set(CUDNN_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/3rdparty/cudnn-8.1.1/include)
set(CUDNN_LIBRARY_PATH ${CMAKE_CURRENT_LIST_DIR}/3rdparty/cudnn-8.1.1/lib/x64)

set(TENSORRT_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tensorrt-7.2.3.4/include)
set(TENSORRT_LIBRARY_PATH ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tensorrt-7.2.3.4/lib)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/opencv-4.5.3/x64/${VC}/staticlib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/opencv-4.5.3/lib/cmake/opencv4)
endif()
set(jsoncpp_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/jsoncpp-1.9.4/lib/cmake/jsoncpp)

cmake_policy(SET CMP0054 NEW)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_package(Threads)
	find_package(Iconv)
endif()

find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)

include_directories(${CUDA_INCLUDE_PATH})
include_directories(${CUDNN_INCLUDE_PATH})
include_directories(${TENSORRT_INCLUDE_PATH})
include_directories(${TENSORRT_INCLUDE_PATH}/../samples/common)
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/jsoncpp-1.9.4/include)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/detectors)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/detectors/yolov5/borrowed)

link_directories(${CUDA_LIBRARY_PATH})
link_directories(${CUDNN_LIBRARY_PATH})
link_directories(${TENSORRT_LIBRARY_PATH})
link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/jsoncpp-1.9.4/lib)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/operators/*.cu
	${CMAKE_CURRENT_LIST_DIR}/src/operators/plugins/*.cu
)

if(WIN32)
	enable_language(CUDA)
endif(WIN32)

add_definitions(-DALGLOG_DEBUG)
add_definitions(-DALGORITHM_DLL)
add_definitions(-DALGORITHM_DLL_EXPORTS)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-arch=compute_75;-code=sm_75)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
endif()


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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

add_subdirectory(model)

cuda_add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} nvinfer cuda cudart ${OpenCV_LIBS} jsoncpp)

add_executable(test-${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/test/main.cpp)
target_link_libraries(test-${PROJECT_NAME} ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_LIST_DIR}/src/algorithm.h")
install(TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(TARGETS test-${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/test/main.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/samples)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config DESTINATION ${CMAKE_INSTALL_PREFIX})