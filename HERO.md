# HERO

HERO is a CPU–GPU inference engine for mixture-of-experts models. It computes cached experts on the GPU and uncached
experts on the CPU, and it can send a share of the uncached experts over PCIe. HERO changes only how that share is
chosen.

This repository contains the engine, the server, setup, and the experiment.

## The HERO split

After each MoE layer's gate output is known, HERO chooses how many uncached experts to transfer to the GPU:

```
i* = argmin over 0 <= i <= N_uc of
     max( a + t_cpu * (N_uc - i),  g0 + t_gpu * N_c + (t_pctr + t_gpu) * i )
```

- `N_c` / `N_uc` are the layer's cached and uncached activated experts.
- `t_cpu`, `t_gpu` and `t_pctr + t_gpu` are per-expert costs, fitted online by least squares to each layer's measured
  CPU and GPU times. `a` and `g0` are fixed per-layer costs.
- The rest of the uncached experts run on the CPU. The decision uses only the current gate output and cache state, with
  no prediction or prefetching.

HERO is on by default. `HERO=0` turns it off and restores the engine's default share; `HERO=0 HERO_PCIE_BALANCE=1`
selects the simpler measured balancer, whose GPU cost has no `N_c` term. HERO adds that term, so a layer with many cache
hits sends fewer misses over PCIe. `HERO_CONTENTION=1` keeps the balancer's DRAM-contention term on the CPU side.

| File                                                  | What it does                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| `include/hero/core/expert_source.hpp`                 | `PcieModel`: the HERO GPU fit `[1, N_c, i]` and the HERO `pick()`               |
| `src/core/expert_source.cpp`                          | counts the layer's cached experts `N_c` and passes them to `pick()`             |
| `include/hero/core/verify.hpp`, `src/core/verify.cpp` | records `N_c` with each layer's timing sample                                   |
| `src/program/generate.cpp`                            | the HERO switch (on by default; `HERO=0` turns it off); logs the fitted costs   |
| `src/core/pcie_hero_test.cpp`                         | unit test: the fit recovers known costs; `pick()` matches a brute-force optimum |
| `hero/bench/`                                         | the experiment driver and its analysis                                          |

## Build and run

Requires Linux, an NVIDIA GPU, CUDA and CMake.

```sh
# unit test of the HERO split (no GPU needed)
g++ -std=c++20 -O1 -Iinclude src/core/pcie_hero_test.cpp -o pcie_hero_test && ./pcie_hero_test

# build the engine from this source and prepare the model. Choose Qwen3.8-Flash-Next,
# ISTA-DASLab IQ3_S (the model used below). Setup prints the run config's path.
./setup.sh --build --no-start

# the engine binary used by the experiment
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHERO_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target hero -j

# the experiment: 5 arms x 3 repetitions x 6 prompts, then the summary
.venv/bin/python hero/bench/ab_driver.py <path to hero-iq3_s.json> build/hero run1 \
    --reps 3 --max-tokens 256 --expert-cache 6400
.venv/bin/python hero/bench/analyze.py run1
```

The driver loads the engine once per (repetition, arm), shuffles the arm order in each repetition, and sends six greedy
prompts of 256 tokens. All arms run the same binary and the same expert-cache size (6,400 of 24,576 experts):

| Arm       | Split of uncached experts                                      |
| --------- | -------------------------------------------------------------- |
| `gpu`     | all transferred to the GPU (GPU offloading)                    |
| `cpu`     | all executed on the CPU                                        |
| `fixed`   | the engine's default share for every layer (`HERO=0`)          |
| `balance` | the measured balancer (`HERO=0 HERO_PCIE_BALANCE=1`)           |
| `hero`    | HERO Timeline-Aware Algorithm (the default; `HERO=1` explicit) |
