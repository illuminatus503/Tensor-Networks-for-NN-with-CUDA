# CC compiler options:
CC=g++
CC_FLAGS= -Wall -g -O2 -std=c++11 -lm
CC_LIBS=

## Project file structure ##
# Source file directory:
SRC_DIR = src
# Object file directory:
OBJ_DIR = bin
# Include header file diretory:
INC_DIR = include

## Make variables ##
# Target executable name:
BIN = run
# Object files: 
SRC_C := $(wildcard $(SRC_DIR)/*.cpp) main.cpp
OBJS_C := $(SRC_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

.PHONY: clean

## Compile ##
# Link c compiled object files to target executable:
$(BIN) : $(OBJS_C) 
	$(CC) -o $@ $(OBJS_C) $(CC_FLAGS)

# Compile main file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CC_FLAGS)

# Handle directories
$(OBJ_DIR):
	mkdir -p $@

# Clean objects in object directory.
clean:
	$(RM) -rv $(BIN) $(OBJ_DIR)

