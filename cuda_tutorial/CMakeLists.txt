cmake_minimum_required(VERSION 3.18)

# ── 1) Make sure ENV{CUDA_HOME} is visible to CMake ────────────────
# If ENV{CUDA_HOME} is empty, we stop immediately.
if (NOT DEFINED ENV{CUDA_HOME})
    message(FATAL_ERROR
            "CUDA_HOME is not set in CMake’s environment. "
            "Before reloading CMake, either:\n"
            "  • Launch CLion from a terminal where you did:\n"
            "      export CUDA_HOME=/usr/local/cuda-12.4\n\n"
            "  • Or go to CLion → Settings → Build, Execution, Deployment → CMake → "
            "and add\n"
            "      CUDA_HOME=/usr/local/cuda-12.4\n"
            "  (adjust path to whatever `which nvcc` shows on your system)."
    )
endif()

# ── 2) Now that we know ENV{CUDA_HOME} exists, force CMake to use its nvcc ──
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_HOME}")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
message(STATUS "→ Using nvcc from: ${CMAKE_CUDA_COMPILER}")

# ── 3) Now declare the project with CUDA support ────────────────────────
project(vector_add LANGUAGES CXX CUDA)

# ── 4) Specify C++17 for device/host code and force “CMake auto-detect GPU arch” ──
set(CMAKE_CUDA_STANDARD      17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# (We do NOT set CMAKE_CUDA_ARCHITECTURES, so CMake will auto-detect.)

# ── 5) Make sure CLion’s indexer can see <cuda_runtime.h> ───────────────
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")

# ── 6) Declare your CUDA executable ──────────────────────────────────────
add_executable(cuda_tut main.cu)
