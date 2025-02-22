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
CMAKE_COMMAND = /snap/cmake/1441/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1441/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build

# Include any dependencies generated for this target.
include CMakeFiles/rtdetr_web.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/rtdetr_web.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/rtdetr_web.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rtdetr_web.dir/flags.make

CMakeFiles/rtdetr_web.dir/codegen:
.PHONY : CMakeFiles/rtdetr_web.dir/codegen

CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/app_rtdetr_web.cpp
CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o -MF CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o.d -o CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/app_rtdetr_web.cpp

CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/app_rtdetr_web.cpp > CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.i

CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/app_rtdetr_web.cpp -o CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.s

CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o: CMakeFiles/rtdetr_web.dir/includes_CUDA.rsp
CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_decode.cu
CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o -MF CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o.d -x cu -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_decode.cu -o CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o

CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_web.cpp
CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o -MF CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o.d -o CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_web.cpp

CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_web.cpp > CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.i

CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/rtdetr_web.cpp -o CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.s

CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/threadpool.cpp
CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o -MF CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o.d -o CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/threadpool.cpp

CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/threadpool.cpp > CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.i

CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/threadpool.cpp -o CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.s

CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/cuda_tools.cpp
CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o -MF CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o.d -o CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/cuda_tools.cpp

CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/cuda_tools.cpp > CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.i

CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/cuda_tools.cpp -o CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.s

CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/ilogger.cpp
CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o -MF CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o.d -o CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/ilogger.cpp

CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/ilogger.cpp > CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.i

CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/ilogger.cpp -o CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.s

CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o: CMakeFiles/rtdetr_web.dir/includes_CUDA.rsp
CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/preprocess_kernel.cu
CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o -MF CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o.d -x cu -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/preprocess_kernel.cu -o CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o

CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_infer.cpp
CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o -MF CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o.d -o CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_infer.cpp

CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_infer.cpp > CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.i

CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_infer.cpp -o CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.s

CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o: CMakeFiles/rtdetr_web.dir/flags.make
CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o: /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_tensor.cpp
CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o: CMakeFiles/rtdetr_web.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o -MF CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o.d -o CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o -c /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_tensor.cpp

CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_tensor.cpp > CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.i

CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/trt_common/trt_tensor.cpp -o CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.s

# Object files for target rtdetr_web
rtdetr_web_OBJECTS = \
"CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o" \
"CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o" \
"CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o" \
"CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o" \
"CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o" \
"CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o" \
"CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o" \
"CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o" \
"CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o"

# External object files for target rtdetr_web
rtdetr_web_EXTERNAL_OBJECTS =

librtdetr_web.so: CMakeFiles/rtdetr_web.dir/apps/app_rtdetr_web.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/apps/rtdetr_decode.cu.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/apps/rtdetr_web.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/apps/threadpool.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/trt_common/cuda_tools.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/trt_common/ilogger.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/trt_common/preprocess_kernel.cu.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/trt_common/trt_infer.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/trt_common/trt_tensor.cpp.o
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/build.make
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
librtdetr_web.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
librtdetr_web.so: CMakeFiles/rtdetr_web.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library librtdetr_web.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rtdetr_web.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rtdetr_web.dir/build: librtdetr_web.so
.PHONY : CMakeFiles/rtdetr_web.dir/build

CMakeFiles/rtdetr_web.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rtdetr_web.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rtdetr_web.dir/clean

CMakeFiles/rtdetr_web.dir/depend:
	cd /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build /data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/build/CMakeFiles/rtdetr_web.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/rtdetr_web.dir/depend

