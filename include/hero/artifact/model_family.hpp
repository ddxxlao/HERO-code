// include/hero/artifact/model_family.hpp - which model family a GGUF holds, from its general.architecture.
//
// HERO runs one model: Qwen3.8-Flash-Next (`qwen4exp`).  The reader, the geometry and the layout tables use this
// header to name a file that is not that model instead of mis-reading it.
//
// No dependencies: the GGUF reader (the architecture guard) and the core geometry both include this.
#pragma once

#include <string>

namespace hero {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,    ///< Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5, the Coder), and quantizations
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp") return ModelFamily::Qwen4Exp;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    return f == ModelFamily::Qwen4Exp ? "qwen4exp" : "";
}

/// The model's name for messages.
inline const char* model_family_title(ModelFamily f) {
    return f == ModelFamily::Qwen4Exp ? "Qwen3.8-Flash-Next" : "an unknown model";
}

/// Whether Flash-Next's engine path runs the family.
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen4Exp; }

/// The refusal for a file this build does not run (empty when it runs it).
inline std::string model_family_not_supported(ModelFamily f) {
    if (model_family_runs(f)) return {};
    return "this is not a model HERO knows; HERO runs Qwen3.8-Flash-Next (architecture 'qwen4exp') files";
}

}  // namespace hero
