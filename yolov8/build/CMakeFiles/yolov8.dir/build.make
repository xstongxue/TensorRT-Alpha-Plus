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
CMAKE_COMMAND = /snap/cmake/1425/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1425/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data02/xs/code/tensorrt-alpha/yolov8

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data02/xs/code/tensorrt-alpha/yolov8/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov8.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yolov8.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov8.dir/flags.make

CMakeFiles/yolov8.dir/codegen:
.PHONY : CMakeFiles/yolov8.dir/codegen

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o: /data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o -MF CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o.d -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o -c /data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp > CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.i

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.s

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o: /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o -MF CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o.d -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o -c /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp > CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.i

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.s

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o: /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp
CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o -MF CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o.d -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o -c /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp > CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.i

CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp -o CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.s

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o: CMakeFiles/yolov8.dir/includes_CUDA.rsp
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o: /data02/xs/code/tensorrt-alpha/utils/kernel_function.cu
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o -MF CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o.d -x cu -c /data02/xs/code/tensorrt-alpha/utils/kernel_function.cu -o CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o: /data02/xs/code/tensorrt-alpha/utils/utils.cpp
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o -MF CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o.d -o CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o -c /data02/xs/code/tensorrt-alpha/utils/utils.cpp

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/tensorrt-alpha/utils/utils.cpp > CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.i

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/tensorrt-alpha/utils/utils.cpp -o CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.s

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o: /data02/xs/code/tensorrt-alpha/utils/yolo.cpp
CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o -MF CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o.d -o CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o -c /data02/xs/code/tensorrt-alpha/utils/yolo.cpp

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/tensorrt-alpha/utils/yolo.cpp > CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.i

CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/tensorrt-alpha/utils/yolo.cpp -o CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.s

CMakeFiles/yolov8.dir/app_yolov8.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/app_yolov8.cpp.o: /data02/xs/code/tensorrt-alpha/yolov8/app_yolov8.cpp
CMakeFiles/yolov8.dir/app_yolov8.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/yolov8.dir/app_yolov8.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/app_yolov8.cpp.o -MF CMakeFiles/yolov8.dir/app_yolov8.cpp.o.d -o CMakeFiles/yolov8.dir/app_yolov8.cpp.o -c /data02/xs/code/tensorrt-alpha/yolov8/app_yolov8.cpp

CMakeFiles/yolov8.dir/app_yolov8.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/app_yolov8.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/tensorrt-alpha/yolov8/app_yolov8.cpp > CMakeFiles/yolov8.dir/app_yolov8.cpp.i

CMakeFiles/yolov8.dir/app_yolov8.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/app_yolov8.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/tensorrt-alpha/yolov8/app_yolov8.cpp -o CMakeFiles/yolov8.dir/app_yolov8.cpp.s

CMakeFiles/yolov8.dir/decode_yolov8.cu.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/decode_yolov8.cu.o: CMakeFiles/yolov8.dir/includes_CUDA.rsp
CMakeFiles/yolov8.dir/decode_yolov8.cu.o: /data02/xs/code/tensorrt-alpha/yolov8/decode_yolov8.cu
CMakeFiles/yolov8.dir/decode_yolov8.cu.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/yolov8.dir/decode_yolov8.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/yolov8.dir/decode_yolov8.cu.o -MF CMakeFiles/yolov8.dir/decode_yolov8.cu.o.d -x cu -c /data02/xs/code/tensorrt-alpha/yolov8/decode_yolov8.cu -o CMakeFiles/yolov8.dir/decode_yolov8.cu.o

CMakeFiles/yolov8.dir/decode_yolov8.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/yolov8.dir/decode_yolov8.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolov8.dir/decode_yolov8.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/yolov8.dir/decode_yolov8.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolov8.dir/yolov8.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/yolov8.cpp.o: /data02/xs/code/tensorrt-alpha/yolov8/yolov8.cpp
CMakeFiles/yolov8.dir/yolov8.cpp.o: CMakeFiles/yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/yolov8.dir/yolov8.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov8.dir/yolov8.cpp.o -MF CMakeFiles/yolov8.dir/yolov8.cpp.o.d -o CMakeFiles/yolov8.dir/yolov8.cpp.o -c /data02/xs/code/tensorrt-alpha/yolov8/yolov8.cpp

CMakeFiles/yolov8.dir/yolov8.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/yolov8.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data02/xs/code/tensorrt-alpha/yolov8/yolov8.cpp > CMakeFiles/yolov8.dir/yolov8.cpp.i

CMakeFiles/yolov8.dir/yolov8.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/yolov8.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data02/xs/code/tensorrt-alpha/yolov8/yolov8.cpp -o CMakeFiles/yolov8.dir/yolov8.cpp.s

# Object files for target yolov8
yolov8_OBJECTS = \
"CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o" \
"CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o" \
"CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o" \
"CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o" \
"CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o" \
"CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o" \
"CMakeFiles/yolov8.dir/app_yolov8.cpp.o" \
"CMakeFiles/yolov8.dir/decode_yolov8.cu.o" \
"CMakeFiles/yolov8.dir/yolov8.cpp.o"

# External object files for target yolov8
yolov8_EXTERNAL_OBJECTS =

libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/logger.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleOptions.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/app/TensorRT-8.6.1.6/samples/common/sampleUtils.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/kernel_function.cu.o
libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/utils.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/data02/xs/code/tensorrt-alpha/utils/yolo.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/app_yolov8.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/decode_yolov8.cu.o
libyolov8.so: CMakeFiles/yolov8.dir/yolov8.cpp.o
libyolov8.so: CMakeFiles/yolov8.dir/build.make
libyolov8.so: /usr/local/cuda/lib64/libcudart_static.a
libyolov8.so: /usr/lib/x86_64-linux-gnu/librt.so
libyolov8.so: /usr/local/cuda/lib64/libcublas.so
libyolov8.so: /usr/local/cuda/lib64/libnppc.so
libyolov8.so: /usr/local/cuda/lib64/libnppig.so
libyolov8.so: /usr/local/cuda/lib64/libnppidei.so
libyolov8.so: /usr/local/cuda/lib64/libnppial.so
libyolov8.so: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvinfer.so
libyolov8.so: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so
libyolov8.so: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvonnxparser.so
libyolov8.so: /data02/xs/app/TensorRT-8.6.1.6/lib/libnvcaffe_parser.so
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
libyolov8.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
libyolov8.so: CMakeFiles/yolov8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library libyolov8.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov8.dir/build: libyolov8.so
.PHONY : CMakeFiles/yolov8.dir/build

CMakeFiles/yolov8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov8.dir/clean

CMakeFiles/yolov8.dir/depend:
	cd /data02/xs/code/tensorrt-alpha/yolov8/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data02/xs/code/tensorrt-alpha/yolov8 /data02/xs/code/tensorrt-alpha/yolov8 /data02/xs/code/tensorrt-alpha/yolov8/build /data02/xs/code/tensorrt-alpha/yolov8/build /data02/xs/code/tensorrt-alpha/yolov8/build/CMakeFiles/yolov8.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/yolov8.dir/depend

