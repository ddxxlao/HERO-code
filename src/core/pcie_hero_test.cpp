// HERO's PcieModel: the fits recover known per-expert costs from synthetic layers, and pick() returns the split
// that minimizes the HERO makespan max(g0 + gv N_c + g i, a + c (N_uc - i)) for every (N_c, N_uc).
#include "hero/core/expert_source.hpp"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
}

// True costs (ms): CPU a + c n_c (+ d n_p), GPU g0 + gv n_v + g n_p.
constexpr double kA = 0.05, kC = 0.12, kD = 0.03, kG0 = 0.08, kGv = 0.01, kG = 0.09;

double brute(int nhit, int nmiss, bool contention, int* best_i) {
    double best = 1e300;
    for (int i = 0; i <= nmiss; ++i) {
        const double cpu = nmiss - i > 0 ? kA + kC * (nmiss - i) + (contention ? kD * i : 0.0) : 0.0;
        const double t = std::max(kG0 + kGv * nhit + kG * i, cpu);
        if (t < best - 1e-6) { best = t; *best_i = i; }
    }
    return best;
}

void run(bool contention) {
    hero::core::PcieModel M;
    M.on = true;
    M.hero = true;
    M.hero_contention = contention;
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.002);
    for (int s = 0; s < 400; ++s) {
        const int nv = (int) (rng() % 11), nmiss = 10 - nv, np = nmiss > 0 ? (int) (rng() % (nmiss + 1)) : 0;
        const int nc = nmiss - np;
        const double cpu = (nc > 0 ? kA + kC * nc : 0.0) + kD * np + noise(rng);
        const double gpu = kG0 + kGv * nv + kG * np + noise(rng);
        M.sample(nc, np, cpu, gpu, nv);
        if (s % 48 == 47) M.solve();
    }
    M.solve();
    check(M.ready(), "ready after 400 layers");
    check(std::fabs(M.c - kC) < 0.01, "t_cpu recovered");
    check(std::fabs(M.gv - kGv) < 0.01, "t_gpu (a hit) recovered");
    check(std::fabs(M.g - kG) < 0.01, "t_pctr + t_gpu recovered");
    int worse = 0;
    for (int nhit = 0; nhit <= 10; ++nhit) {
        const int nmiss = 10 - nhit;
        int want = 0;
        const double opt = brute(nhit, nmiss, contention, &want);
        M.explore = 1;   // keep the every-64th exploration step out of this check
        const int got = M.pick(nmiss, 0, nhit);
        const double cpu = nmiss - got > 0 ? kA + kC * (nmiss - got) + (contention ? kD * got : 0.0) : 0.0;
        const double t = std::max(kG0 + kGv * nhit + kG * got, cpu);
        if (t > opt * 1.05 + 1e-6) ++worse;
    }
    check(worse == 0, "pick within 5% of the brute-force HERO makespan for every N_c");
    // N_c moves the split: more hits load the GPU, so fewer misses should go over PCIe
    M.explore = 1;
    const int i_few_hits = M.pick(8, 0, 2);
    M.explore = 1;
    const int i_many_hits = M.pick(8, 0, 40);
    check(i_many_hits <= i_few_hits, "more VRAM hits never send more misses over PCIe");
    std::printf("%s: a %.3f c %.3f d %.3f | g0 %.3f gv %.3f g %.3f | split(N_c=2,N_uc=8)=%d split(N_c=40,N_uc=8)=%d\n",
                contention ? "contention" : "hero", M.a, M.c, M.d, M.g0, M.gv, M.g, i_few_hits, i_many_hits);
}

}  // namespace

int main() {
    run(false);
    run(true);
    std::printf(fails == 0 ? "pcie_hero_test: OK\n" : "pcie_hero_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
