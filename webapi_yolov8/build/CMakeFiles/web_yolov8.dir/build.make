# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1429/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1429/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build

# Include any dependencies generated for this target.
include CMakeFiles/web_yolov8.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/web_yolov8.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/web_yolov8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/web_yolov8.dir/flags.make

CMakeFiles/web_yolov8.dir/codegen:
.PHONY : CMakeFiles/web_yolov8.dir/codegen

CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o: CMakeFiles/web_yolov8.dir/flags.make
CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/web_yolov8.cpp
CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o: CMakeFiles/web_yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o -MF CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o.d -o CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/web_yolov8.cpp

CMakeFiles/web_yolov8.dir/web_yolov8.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/web_yolov8.dir/web_yolov8.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/web_yolov8.cpp > CMakeFiles/web_yolov8.dir/web_yolov8.cpp.i

CMakeFiles/web_yolov8.dir/web_yolov8.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/web_yolov8.dir/web_yolov8.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/web_yolov8.cpp -o CMakeFiles/web_yolov8.dir/web_yolov8.cpp.s

# Object files for target web_yolov8
web_yolov8_OBJECTS = \
"CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o"

# External object files for target web_yolov8
web_yolov8_EXTERNAL_OBJECTS =

web_yolov8: CMakeFiles/web_yolov8.dir/web_yolov8.cpp.o
web_yolov8: CMakeFiles/web_yolov8.dir/build.make
web_yolov8: libyolov8.so
web_yolov8: /usr/local/cuda/lib64/libcudart_static.a
web_yolov8: /usr/lib/x86_64-linux-gnu/librt.so
web_yolov8: /usr/local/cuda/lib64/libcublas.so
web_yolov8: /usr/local/cuda/lib64/libnppc.so
web_yolov8: /usr/local/cuda/lib64/libnppig.so
web_yolov8: /usr/local/cuda/lib64/libnppidei.so
web_yolov8: /usr/local/cuda/lib64/libnppial.so
web_yolov8: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvinfer.so
web_yolov8: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so
web_yolov8: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvonnxparser.so
web_yolov8: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvcaffe_parser.so
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
web_yolov8: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
web_yolov8: CMakeFiles/web_yolov8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable web_yolov8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/web_yolov8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/web_yolov8.dir/build: web_yolov8
.PHONY : CMakeFiles/web_yolov8.dir/build

CMakeFiles/web_yolov8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/web_yolov8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/web_yolov8.dir/clean

CMakeFiles/web_yolov8.dir/depend:
	cd /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8 /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8 /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_yolov8/build/CMakeFiles/web_yolov8.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/web_yolov8.dir/depend

