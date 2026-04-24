# XasmAI Makefile
# C++20 CPU + CUDA GPU Build System

CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -I include -march=native -fopenmp
LDFLAGS = -fopenmp

NVCC = nvcc
CUDA_ARCH ?= sm_75
CUDA_FLAGS = -arch=$(CUDA_ARCH) --use_fast_math -O3 -std=c++20 -I include
CUDA_LDFLAGS = -lcudart -lcublas

# Hedefler
.PHONY: all clean xor regression agent gan gpt gpu gpt_gpu gpt_cuda run_xor run_regression run_agent run_gan run_gpt run_gpt_gpu run_gpt_cuda help

help:
	@echo ""
	@echo "  ╔═══════════════════════════════════════════╗"
	@echo "  ║         XasmAI Build System               ║"
	@echo "  ╠═══════════════════════════════════════════╣"
	@echo "  ║  CPU Targets                              ║"
	@echo "  ╠═══════════════════════════════════════════╣"
	@echo "  ║  make all           — Tüm CPU örnekleri   ║"
	@echo "  ║  make xor           — XOR demo            ║"
	@echo "  ║  make regression    — Regression demo     ║"
	@echo "  ║  make agent         — RL Agent            ║"
	@echo "  ║  make gan           — GAN Loop            ║"
	@echo "  ║  make gpt           — GPT Transformer     ║"
	@echo "  ╠═══════════════════════════════════════════╣"
	@echo "  ║  GPU Targets (CUDA + FP16)                ║"
	@echo "  ╠═══════════════════════════════════════════╣"
	@echo "  ║  make gpt_cuda      — GPU Transformer      ║"
	@echo "  ║  make run_gpt_cuda  — GPU GPT çalıştır    ║"
	@echo "  ║                                            ║"
	@echo "  ║  RTX 2060: make gpt_cuda CUDA_ARCH=sm_75  ║"
	@echo "  ║  RTX 30xx: make gpt_cuda CUDA_ARCH=sm_86  ║"
	@echo "  ║  RTX 40xx: make gpt_cuda CUDA_ARCH=sm_89  ║"
	@echo "  ║  RTX 50xx: make gpt_cuda CUDA_ARCH=sm_120 ║"
	@echo "  ╠═══════════════════════════════════════════╣"
	@echo "  ║  make clean         — Temizle             ║"
	@echo "  ╚═══════════════════════════════════════════╝"
	@echo ""
	@echo "  GPU Arch: $(CUDA_ARCH) (override: make CUDA_ARCH=sm_89 gpt_gpu)"
	@echo ""

all: xor regression agent gan gpt

# ====== XOR Demo ======
xor: examples/xor_demo.cpp
	@echo "  🔨 Derleniyor: XOR Demo..."
	$(CXX) $(CXXFLAGS) examples/xor_demo.cpp -o xor_demo $(LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./xor_demo"

run_xor: xor
	@echo ""
	@./xor_demo

# ====== Regression Demo ======
regression: examples/regression_demo.cpp
	@echo "  🔨 Derleniyor: Regression Demo..."
	$(CXX) $(CXXFLAGS) examples/regression_demo.cpp -o regression_demo $(LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./regression_demo"

run_regression: regression
	@echo ""
	@./regression_demo

# ====== RL Agent ======
agent: examples/continuous_loop.cpp
	@echo "  🔨 Derleniyor: RL Agent..."
	$(CXX) $(CXXFLAGS) examples/continuous_loop.cpp -o continuous_loop $(LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./continuous_loop"

run_agent: agent
	@echo ""
	@./continuous_loop

# ====== GAN Loop ======
gan: examples/gan_loop.cpp
	@echo "  🔨 Derleniyor: GAN Loop..."
	$(CXX) $(CXXFLAGS) examples/gan_loop.cpp -o gan_loop $(LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./gan_loop"

run_gan: gan
	@echo ""
	@./gan_loop

# ====== GPT Transformer (CPU) ======
gpt: examples/gpt_loop.cpp
	@echo "  🔨 Derleniyor: GPT Transformer (CPU)..."
	$(CXX) $(CXXFLAGS) examples/gpt_loop.cpp -o gpt_loop $(LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./gpt_loop <metin_dosyasi>"

run_gpt: gpt
	@echo ""
	@./gpt_loop data.txt

# ====== GPU Transformer (CUDA + cuBLAS) ======
gpt_cuda: cuda/gpt_cuda.cu
	@echo "  🔥 Derleniyor: GPU Transformer ($(CUDA_ARCH), cuBLAS Tensor Core)..."
	$(NVCC) $(CUDA_FLAGS) cuda/gpt_cuda.cu -o gpt_cuda $(CUDA_LDFLAGS)
	@echo "  ✅ Derleme başarılı: ./gpt_cuda <metin_dosyasi>"

run_gpt_cuda: gpt_cuda
	@echo ""
	@./gpt_cuda data.txt

# ====== Temizle ======
clean:
	@echo "  🧹 Temizleniyor..."
	rm -f xor_demo regression_demo continuous_loop gan_loop gpt_loop gpt_gpu gpt_cuda *.xasm cuda/*.o
	@echo "  ✅ Temiz!"
