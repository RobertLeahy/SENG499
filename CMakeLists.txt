cmake_minimum_required(VERSION 2.6)
set(CMAKE_OSX_SYSROOT /)
set(CMAKE_MACOSX_RPATH 1)
project(DynFu LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CheckIncludeFileCXX)

function(add_linker_options)
	foreach(arg ${ARGV})
		set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} ${arg}")
		set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES}" PARENT_SCOPE)
	endforeach(arg)
endfunction()

#	Check for Filesystem TS
set(CMAKE_REQUIRED_FLAGS -std=c++1z)
check_include_file_cxx(experimental/filesystem HAS_FILESYSTEM_TS)
if (HAS_FILESYSTEM_TS)
	add_linker_options(-lstdc++fs)
else()
	add_linker_options(-lboost_filesystem -lboost_system)
endif()

#	Where our project's headers live
include_directories(include)

#	General command line arguments to the compiler that should be present
#	on all platforms
add_compile_options(-std=c++1z)
if(DEFINED RELEASE AND RELEASE)
	add_compile_options(-O2)
else()
	add_compile_options(-O0 -g -fno-inline -fno-omit-frame-pointer -Wall -Wpedantic -Wextra -Werror)
	#	Address Sanitizer (non-Windows only)
	if(NOT WIN32)
		#	Address Sanitizer needs to be able to be turned off due
		#	to the fact the AMD OpenCL CPU SDK doesn't play nice with
		#	it
		if(NOT DEFINED USE_ADDRESS_SANITIZER OR USE_ADDRESS_SANITIZER)
			add_compile_options(-fsanitize=address)
			add_linker_options(-fsanitize=address)
		endif()
	endif()
	#	Suppress some of Clang's excessive warnings
	if (DEFINED CMAKE_CXX_COMPILER_ID AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		add_compile_options(-Wno-unused-private-field)
	endif()
endif()

#	OpenCL and other miscellaneous frameworks for libigl
if(APPLE)
	add_linker_options(-framework OpenCL -framework CoreVideo -framework IOKit -framework Cocoa)
else()
	add_linker_options(-lOpenCL)
endif()

#	OpenCL include directory on Windows (this is hardcoded and kind of
#	a hack)
if(WIN32)
	include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5\\include")
	link_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5\\lib\\x64")
endif()

#	libigl-related code won't assemble on Windows without this (too many sections)
#	It also seems to matter that this come before add_executable (doesn't
#	show up if it comes after like everything else)
if (WIN32)
	add_compile_options("-Wa,-mbig-obj")
endif()
#	Try and workaround THIS issue with libigl and Windows headers
#	The issue: The libigl source uses THIS as a variable, whereas
#	certain Windows headers #define it to be something else
#	These offending Windows headers aren't pulled in with
#	WIN32_LEAN_AND_MEAN
if (WIN32)
	add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

add_library(dynfu SHARED
	src/camera.cpp
	src/depth_device.cpp
	src/depth_device_decorator.cpp
	src/file_system_depth_device.cpp
	src/file_system_opencl_program_factory.cpp
	src/fps_depth_device.cpp
	src/high_resolution_clock.cpp
	src/kinect_fusion.cpp
	src/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.cpp
	src/kinect_fusion_opencl_measurement_pipeline_block.cpp
	src/kinect_fusion_opencl_pose_estimation_pipeline_block.cpp
	src/kinect_fusion_opencl_surface_prediction_pipeline_block.cpp
	src/kinect_fusion_opencl_update_reconstruction_pipeline_block.cpp
	src/measurement_pipeline_block.cpp
	src/mock_depth_device.cpp
	src/msrc_file_system_depth_device.cpp
	src/opencl_build_error.cpp
	src/opencl_depth_device.cpp
	src/opencl_program_factory.cpp
	src/opencv_depth_device.cpp
	src/path.cpp
	src/pose_estimation_pipeline_block.cpp
	src/surface_prediction_pipeline_block.cpp
	src/update_reconstruction_pipeline_block.cpp
	src/whereami.cpp
)

add_library(dynfu_libigl SHARED src/libigl.cpp)
find_package(OpenGL REQUIRED)
target_link_libraries(dynfu_libigl ${OPENGL_gl_LIBRARY} pthread)
#	We link against GLFW using a DLL in bin on Windows so this flag
#	isn't needed (and actually breaks everything)
if (NOT WIN32)
	target_link_libraries(dynfu_libigl glfw)
endif()
if (NOT APPLE AND NOT WIN32)
	find_package(GLEW REQUIRED)
	target_link_libraries(dynfu_libigl ${GLEW_LIBRARIES})
endif()
#	Special GLEW/GLFW linker nonsense on Windows
if (WIN32)
	set_target_properties(dynfu_libigl PROPERTIES LINK_FLAGS "bin/glew32.dll bin/glfw3.dll bin/*opencv*.dll")
endif()

add_executable(client src/main.cpp)
target_link_libraries(client dynfu dynfu_libigl boost_program_options)

add_executable(tests
	src/test/camera.cpp
	src/test/cpu_pipeline_value.cpp
	src/test/file_system_depth_device.cpp
	src/test/file_system_opencl_program_factory.cpp
	src/test/fps_depth_device.cpp
	src/test/kinect_fusion.cpp
	src/test/kinect_fusion_eigen_pose_estimation_pipeline_block.cpp
	src/test/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.cpp
	src/test/kinect_fusion_opencl_measurement_pipeline_block.cpp
	src/test/kinect_fusion_opencl_pose_estimation_pipeline_block.cpp
	src/test/kinect_fusion_opencl_surface_prediction_pipeline_block.cpp
	src/test/kinect_fusion_opencl_update_reconstruction_pipeline_block.cpp
	src/test/main.cpp
	src/test/msrc_file_system_depth_device.cpp
	src/test/opencl_build_error.cpp
	src/test/opencl_depth_device.cpp
	src/test/opencl_pipeline_value.cpp
	src/test/opencl_vector_pipeline_value.cpp
	src/test/opencv_depth_device.cpp
)
target_link_libraries(tests dynfu)

add_executable(tsdf_viewer
	src/tsdf_viewer.cpp
)
target_link_libraries(tsdf_viewer dynfu dynfu_libigl)

add_custom_target(tests_run ALL
	COMMAND tests
	DEPENDS tests
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	COMMENT "Run test suite"
)

#	Special OpenCV linker nonsense
if(WIN32)
	set_target_properties(dynfu tests client PROPERTIES LINK_FLAGS "bin/*opencv*.dll")
else()
	execute_process(COMMAND pkg-config --libs opencv OUTPUT_VARIABLE OPENCV_PKGCONFIG OUTPUT_STRIP_TRAILING_WHITESPACE)
	add_linker_options(${OPENCV_PKGCONFIG})
endif()