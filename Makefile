## C-COMPILER ##
# CC compiler options:
CC=g++
CC_FLAGS= -Wall -O3
CC_LIBS=

## NVCC ##
# CUDA directory:
CUDA_ROOT_DIR=/usr
# NVCC compiler options:
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS= -O3
NVCC_LIBS=
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

## PROJECT ##
# Target executable name:
BIN = par23

## Project file structure ##
# Source file directory:
SRC = src
# Object file directory:
OBJ = bin
# Include header file diretory:
INC = include

# CU files
SRC_CU := $(wildcard $(SRC)/*.cu)

# C files
SRC_C := $(wildcard $(SRC)/*.c) main.c

# OBJ files
OBJS_C := $(SRC_C:$(SRC)/%.c=$(OBJ)/%.o) $(SRC_CU:$(SRC)/%.cu=$(OBJ)/%.o)

## Compile ##
.PHONY: all clean
all: $(BIN)

# Link c and CUDA compiled object files to target executable:
$(BIN) : $(OBJS_C) | $(SRC)
	$(CC) $(CC_FLAGS) $(OBJS_C) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main.c file to object files:
$(OBJ)/%.o : $(SRC)/%.c | $(OBJ)
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C source files to object files:
$(OBJ)/%.o : $(SRC)/%.c $(INC)/%.h | $(OBJ)
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ)/%.o : $(SRC)/%.cu $(INC)/%.cuh | $(OBJ)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Handle directories
$(OBJ):
	mkdir -p $@

# Clean objects in object directory.
clean:
	$(RM) -rv $(OBJ) $(BIN)
