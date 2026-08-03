CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp -I src
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_89 -I src -Xcompiler -fPIC

HAS_NVCC := $(shell which nvcc 2>/dev/null)

ifneq ($(HAS_NVCC),)
    CXXFLAGS += -DUSE_CUDA -I/usr/local/cuda/include -I/usr/include/cuda
    NVCCFLAGS += -DUSE_CUDA
    CUDA_SRCS = $(shell find src -name "*.cu")
    CUDA_OBJS = $(patsubst src/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRCS))
    CUDA_LIBS = -L/usr/lib/x86_64-linux-gnu -lcublas -lcudart
else
    CUDA_SRCS =
    CUDA_OBJS =
    CUDA_LIBS =
endif

TARGET = matmul_free_llm
BUILD_DIR = build

CPP_SRCS = $(shell find src -name "*.cpp")
CPP_OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))

OBJS = $(CPP_OBJS) $(CUDA_OBJS)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CUDA_LIBS)

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
