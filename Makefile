# Makefile for CUDAXPBD project

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = -std=c++20 -arch=sm_80 -rdc=true

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES))

# Include directories
INCLUDES = -I$(INCLUDE_DIR)

# Target shared library
TARGET_LIB = $(BUILD_DIR)/libCUDAXPBD.so

# Target executable
TARGET_EXEC = $(BUILD_DIR)/CUDAXPBD_executable

# Rules
all: $(TARGET_LIB) $(TARGET_EXEC)

$(TARGET_LIB): $(OBJECTS)
	$(NVCC) $(CFLAGS) -shared -o $@ $^

$(TARGET_EXEC): $(OBJECTS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJECTS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Phony targets
.PHONY: all clean

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET_LIB) $(TARGET_EXEC)
