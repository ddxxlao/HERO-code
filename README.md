# HERO

HERO is a per-layer expert scheduler for CPU–GPU mixture-of-experts inference. Cached experts run on the GPU, uncached
experts run on the CPU, and after each layer's gate HERO chooses how many of the uncached experts to transfer over PCIe
(the Timeline-Aware Algorithm).

This repository contains the engine, the server, setup, and the experiment. See [HERO.md](HERO.md) for the change and the
build and run steps.

Requires Linux, an NVIDIA GPU, CUDA and CMake. License: MIT (`LICENSE`).
