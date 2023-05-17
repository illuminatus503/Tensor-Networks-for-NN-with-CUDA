# CUDA directory:
CUDA_ROOT_DIR=/usr

# CC compiler options:
CC=g++
CC_FLAGS= -Wall -O3
CC_LIBS=

# NVCC compiler options:
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64

# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include

# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

## Project file structure ##
# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

## Make variables ##
# Target executable name:
BIN = test

#! Object files: (!!Añadir aquí cada nuevo fichero fuente!!)
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/tq_matrix.o $(OBJ_DIR)/__tq_op_cpu.o $(OBJ_DIR)/__tq_op_gpu.o $(OBJ_DIR)/tq_perceptron.o

## Compile ##
# Link c and CUDA compiled object files to target executable:
$(BIN) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main.c file to object files:
$(OBJ_DIR)/%.o : %.c
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/*.o $(BIN)
