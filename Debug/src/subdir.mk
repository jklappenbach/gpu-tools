################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/AxisAlignedBoundingBox.cpp \
../src/BoundingVolumeHierarchy.cpp \
../src/QuickSort.cpp \
../src/Time.cpp \
../src/main.cpp 

CU_SRCS += \
../src/BinaryRadixTree.cu \
../src/BitonicSort.cu \
../src/BoundingVolumeHiearchy.cu \
../src/ParallelPrefixSum.cu \
../src/RadixSort.cu 

CU_DEPS += \
./src/BinaryRadixTree.d \
./src/BitonicSort.d \
./src/BoundingVolumeHiearchy.d \
./src/ParallelPrefixSum.d \
./src/RadixSort.d 

OBJS += \
./src/AxisAlignedBoundingBox.o \
./src/BinaryRadixTree.o \
./src/BitonicSort.o \
./src/BoundingVolumeHiearchy.o \
./src/BoundingVolumeHierarchy.o \
./src/ParallelPrefixSum.o \
./src/QuickSort.o \
./src/RadixSort.o \
./src/Time.o \
./src/main.o 

CPP_DEPS += \
./src/AxisAlignedBoundingBox.d \
./src/BoundingVolumeHierarchy.d \
./src/QuickSort.d \
./src/Time.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/julian/cuda-workspace/game-engine/include" -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/julian/cuda-workspace/game-engine/include" -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/julian/cuda-workspace/game-engine/include" -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/julian/cuda-workspace/game-engine/include" -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


