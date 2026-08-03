CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp -I src
TARGET = matmul_free_llm
BUILD_DIR = build

SRCS = $(shell find src -name "*.cpp")
OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
