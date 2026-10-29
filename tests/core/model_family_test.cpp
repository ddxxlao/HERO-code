// tests/core/model_family_test.cpp - the model family (qwen4exp) on synthetic GGUFs: no model, no GPU.
//
//   * the architecture guard accepts qwen4exp and refuses anything else; the engine's guard runs qwen4exp;
//   * geometry_from_gguf reads Flash-Next's metadata as exactly the compiled defaults;
//   * the layout tables on tiny files of the family: a complete file passes, a wrong shape and a missing tensor
//     are refused, naming the tensor.
//
// `model_family_test --real SHARD`: the same view of a real model (headers only): family, geometry, the kernels'
// verdict and the layout check over every shard.
#include "gguf_fixture.hpp"

#include "hero/core/family_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("hero-model-family-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, hc, hc_lr, idx_heads, idx_key;
};
// Qwen3.8-Flash-Next, as its files say
const Dims kFlashNext = {48, 2560, 24, 2, 256, 512, 10, 640, 640, 4, 128, 16, 48, 6144, 4, 4, 320, 4, 128};
// a tiny model for the layout checks (every tensor F32, a few KB)
const Dims kTinyFlash = {8, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 2, 8, 2, 8};

std::vector<Kv> metadata(const char* arch, const Dims& d, uint64_t vocab) {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_feed_forward_length", d.ff_sh),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval),
                          fixture::u32(p + "hyper_connection.count", d.hc),
                          fixture::u32(p + "hyper_connection.low_rank", d.hc_lr),
                          fixture::u32(p + "attention.indexer.head_count", d.idx_heads),
                          fixture::u32(p + "attention.indexer.key_length", d.idx_key)};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// The tensors the layout table of `g` names, as F32 fixtures.
std::vector<Tensor> tensors_of(const hero::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, F32, seed++});
    };
    for (const auto& w : hero::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : hero::core::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

bool geometry_of(const fs::path& p, hero::core::ModelGeometry& g, int64_t& k, std::string& err) {
    const hero::GgufFile f(p.string());
    return hero::core::geometry_from_gguf(f, g, k, err);
}

std::string layout_error(const fs::path& dir, const char* arch, const Dims& d, uint64_t vocab,
                         void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "meta.gguf";
    fixture::write(meta, metadata(arch, d, vocab), {});
    hero::core::ModelGeometry g;
    std::string err;
    // the qwen4exp guard pins Flash-Next's own block count and widths, so the tiny model's geometry is set here
    g.n_layers = (int64_t) d.block_count;
    g.n_embd = (int64_t) d.embd;
    g.n_head = (int64_t) d.heads;
    g.n_head_kv = (int64_t) d.heads_kv;
    g.head_dim = (int64_t) d.key_len;
    g.n_expert = (int64_t) d.experts;
    g.n_ff = (int64_t) d.ff;
    g.n_ff_shexp = (int64_t) d.ff_sh;
    g.ssm_d_conv = (int64_t) d.conv;
    g.ssm_state_size = (int64_t) d.state;
    g.ssm_k_heads = (int64_t) d.groups;
    g.ssm_v_heads = (int64_t) d.v_heads;
    g.ssm_value_dim = (int64_t) d.inner;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    g.hc = (int64_t) d.hc;
    g.hc_lr = (int64_t) d.hc_lr;
    g.idx_q_heads = (int64_t) d.idx_heads;
    g.idx_key_dim = (int64_t) d.idx_key;
    g.n_vocab = (int64_t) vocab;
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / (std::string(arch) + "-tiny.gguf");
    fixture::write(model, metadata(arch, d, vocab), ts);
    const hero::GgufModel m({model.string()});
    size_t n = 0;
    if (!hero::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const hero::GgufModel m = hero::GgufModel::open(any);
        const std::string arch = hero::check_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s; engine guard: %s\n", m.size(),
                    arch.empty() ? "ok" : arch.c_str(),
                    hero::check_engine_architecture(m.meta()).empty() ? "runs"
                                                                         : hero::check_engine_architecture(m.meta()).c_str());
        if (!arch.empty()) return 1;
        hero::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        if (!hero::core::geometry_from_gguf(m.meta(), g, k, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  family %s: layers %lld (+%lld MTP), hidden %lld, attention %lld/%lld x %lld every %lld, GDN K %lld"
                    " V %lld x %lld conv %lld (%lld), MoE %lld experts top-%lld FF %lld shared %lld, hc %lld, indexer"
                    " %lld, vocab %lld\n",
                    hero::model_family_title(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers,
                    (long long) g.n_embd, (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim,
                    (long long) g.qsa_interval, (long long) g.ssm_k_heads, (long long) g.ssm_v_heads,
                    (long long) g.ssm_state_size, (long long) g.ssm_conv_channels, (long long) g.ssm_d_conv,
                    (long long) g.n_expert, (long long) k, (long long) g.n_ff, (long long) g.n_ff_shexp,
                    (long long) g.hc, (long long) g.idx_q_heads, (long long) g.n_vocab);
        const std::string mm = hero::core::compiled_geometry_mismatch(g);
        std::printf("  compiled kernels: %s\n", mm.empty() ? "match" : mm.c_str());
        size_t n = 0;
        const bool ok = hero::core::check_gguf_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("model_family_test\n");
    TempDir d;
    {
        // the guards
        const fs::path fn = d.path / "fn.gguf", other = d.path / "other.gguf";
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 8), {});
        fixture::write(other, {fixture::str("general.architecture", "llama")}, {});
        const hero::GgufFile b(fn.string()), c(other.string());
        check(hero::model_family(b) == hero::ModelFamily::Qwen4Exp &&
                  hero::model_family(c) == hero::ModelFamily::Unknown,
              "model_family: qwen4exp, unknown");
        check(hero::check_architecture(b).empty(), "the architecture guard accepts qwen4exp");
        check(hero::check_architecture(c).find("'llama'") != std::string::npos, "and refuses llama, naming it");
        check(hero::check_engine_architecture(b).empty(), "the engine guard runs qwen4exp");
    }
    {
        // the geometry of the real model, from its metadata values
        const fs::path fn = d.path / "fng.gguf";
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 248320), {});
        hero::core::ModelGeometry f;
        int64_t k = 0;
        std::string err;
        const bool okf = geometry_of(fn, f, k, err);
        const hero::core::ModelGeometry c;
        check(okf && hero::core::compiled_geometry_mismatch(f).empty() && k == 10 && f.n_expert == c.n_expert &&
                  f.ssm_conv_channels == c.ssm_conv_channels && f.n_vocab == c.n_vocab && f.hc_dim() == c.hc_dim(),
              "Flash-Next's metadata gives exactly the compiled geometry" + (okf ? std::string() : ": " + err));
        Dims coder = kFlashNext;
        coder.experts = 256;
        const fs::path cp = d.path / "coder.gguf";
        fixture::write(cp, metadata("qwen4exp", coder, 8), {});
        check(geometry_of(cp, f, k, err) && f.n_expert == 256 && hero::core::compiled_geometry_mismatch(f).empty(),
              "the Coder's 256 experts are a run-time size, not a kernel mismatch");
        Dims wrong = kFlashNext;
        wrong.inner = 4000;
        const fs::path wp = d.path / "wrong.gguf";
        fixture::write(wp, metadata("qwen4exp", wrong, 8), {});
        check(!geometry_of(wp, f, k, err) && err.find("ssm.inner_size") != std::string::npos,
              "an inconsistent ssm.inner_size is refused");
    }
    {
        // the layout tables
        const std::string ef = layout_error(d.path, "qwen4exp", kTinyFlash, 16, nullptr);
        check(ef.empty(), "a complete qwen4exp file passes its layout (hyper-connections, indexer)" +
                              (ef.empty() ? std::string() : ": " + ef));
        std::string e = layout_error(d.path, "qwen4exp", kTinyFlash, 16, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.1.ssm_conv1d.weight") t.shape[1] += 1;
        });
        check(e.find("blk.1.ssm_conv1d.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong shape is refused, naming the tensor and both shapes");
        e = layout_error(d.path, "qwen4exp", kTinyFlash, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.indexer.k_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.indexer.k_proj.weight", "a missing indexer tensor is refused (" + e + ")");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
