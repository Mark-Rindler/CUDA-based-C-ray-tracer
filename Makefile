# ================= CUDA + MSVC + GLAD + GLFW (Windows) =================
# Usage:
#   make                 # release build
#   make CONFIG=debug    # debug build
#
# You can override these on the command line:
#   make VCVARS="C:\Path\to\vcvarsall.bat" CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"

SHELL := cmd

# ---- Paths -------------------------------------------------------------
PROJECT_DIR := .
SRC_DIR     := src
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := bin

GLAD_ROOT   := libs/glad
GLAD_INC    := $(GLAD_ROOT)/include

# Try GLAD v2 (gl.c) first, fall back to GLAD v1 (glad.c)
ifneq ("$(wildcard $(GLAD_ROOT)/src/gl.c)","")
  GLAD_SRC := $(GLAD_ROOT)/src/gl.c
else
  GLAD_SRC := $(GLAD_ROOT)/src/glad.c
endif

GLFW_ROOT   := libs/glfw-3.4.bin.WIN64
GLFW_INC    := $(GLFW_ROOT)/include
GLFW_LIBDIR := $(GLFW_ROOT)/lib-vc2022

# CUDA
CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0
NVCC      ?= "$(CUDA_PATH)/bin/nvcc.exe"

# Visual Studio vcvars (override if your edition/path differs)
VCVARS ?= C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat

# ---- Target ------------------------------------------------------------
TARGET := raytracer.exe

# ---- Config (release/debug) -------------------------------------------
CONFIG ?= release

# Common flags
DEFS      := /DWIN32_LEAN_AND_MEAN /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DGLFW_INCLUDE_NONE
INCLUDES  := -I"$(GLFW_INC)" -I"$(GLAD_INC)" -I"$(SRC_DIR)"

# SM architecture (override ARCH if you want a different one)
ARCH ?= -gencode arch=compute_86,code=sm_86

ifeq ($(CONFIG),debug)
  CLFLAGS    := /nologo /MDd /Zi /Od $(DEFS)
  NVCCFLAGS  := -std=c++17 -G -Xcompiler="/MDd /Zi" $(INCLUDES)
  LDFLAGS    := /DEBUG
else
  CLFLAGS    := /nologo /MD /O2 $(DEFS)
  NVCCFLAGS  := -std=c++17 -O2 -Xcompiler="/MD" $(INCLUDES)
  LDFLAGS    :=
endif

# ---- Source / Object discovery ----------------------------------------
# All .cu files in src/ (top-level). Add more pattern rules if you need subdirs.
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS    := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.obj)

# GLAD object name
GLAD_OBJ := $(OBJ_DIR)/glad.obj

# ---- Helpers: Windows path conversion ---------------------------------
# $(call wpath, forward/slash/path) -> backslash\path (for cl/link on cmd.exe)
wpath = $(subst /,\,$1)

BUILD_DIR_WIN := $(call wpath,$(BUILD_DIR))
OBJ_DIR_WIN   := $(call wpath,$(OBJ_DIR))
BIN_DIR_WIN   := $(call wpath,$(BIN_DIR))
GLFW_LIBDIR_WIN := $(call wpath,$(GLFW_LIBDIR))

# ---- Phony -------------------------------------------------------------
.PHONY: all clean dirs

all: $(BIN_DIR)/$(TARGET)

# Ensure dirs exist
dirs:
	@if not exist "$(BUILD_DIR_WIN)" mkdir "$(BUILD_DIR_WIN)"
	@if not exist "$(OBJ_DIR_WIN)"   mkdir "$(OBJ_DIR_WIN)"
	@if not exist "$(BIN_DIR_WIN)"   mkdir "$(BIN_DIR_WIN)"

# ---- Compile GLAD (with cl) -------------------------------------------
$(GLAD_OBJ): $(GLAD_SRC) | dirs
	@echo Compiling GLAD $(notdir $<)...
	@call "$(VCVARS)" x64 >nul && cl $(CLFLAGS) /I"$(call wpath,$(GLAD_INC))" /I"$(call wpath,$(GLFW_INC))" /c "$(call wpath,$<)" /Fo"$(call wpath,$@)"

# ---- Compile CUDA (.cu -> .obj with nvcc) ------------------------------
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu | dirs
	@echo Compiling CUDA $<
	@call "$(VCVARS)" x64 >nul && $(NVCC) $(NVCCFLAGS) $(ARCH) -c "$(call wpath,$<)" -o "$(call wpath,$@)"

# ---- Link (MSVC link.exe) ---------------------------------------------
$(BIN_DIR)/$(TARGET): $(OBJECTS) $(GLAD_OBJ) | dirs
	@echo Linking $@ ...
	@call "$(VCVARS)" x64 >nul && link /nologo /OUT:"$(call wpath,$@)" \
		$(foreach o,$(OBJECTS) $(GLAD_OBJ),"$(call wpath,$(o))") \
		/LIBPATH:"$(GLFW_LIBDIR_WIN)" /LIBPATH:"$(call wpath,$(CUDA_PATH))/lib/x64" \
		glfw3dll.lib opengl32.lib user32.lib gdi32.lib shell32.lib cudart.lib $(LDFLAGS)
	@rem Copy the DLL so the exe runs without PATH tweaks
	@if exist "$(GLFW_LIBDIR_WIN)\glfw3.dll" copy /Y "$(GLFW_LIBDIR_WIN)\glfw3.dll" "$(BIN_DIR_WIN)" >nul

# ---- Clean -------------------------------------------------------------
clean:
	@if exist "$(BUILD_DIR_WIN)" rmdir /S /Q "$(BUILD_DIR_WIN)"
	@rem Keep bin/ but remove exe
	@if exist "$(BIN_DIR_WIN)\$(TARGET)" del /Q "$(BIN_DIR_WIN)\$(TARGET)"
