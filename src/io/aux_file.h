/*
 * src/io/aux_file.h - locate a model's auxiliary file.
 *
 * Shared by the engine (tokenizer.bin) and the encoder towers
 * (audio_tower.safetensors, mel_constants.bin, vision_tower.safetensors).
 */
#ifndef GEIST_AUX_FILE_H
#define GEIST_AUX_FILE_H

/* The first readable one of: $env_name (when env_name is set and the
 * variable is non-empty), <dir>/<basename> (when dir is set), then each
 * entry of the nullptr-terminated `fallbacks` (relative to the working
 * directory; may be nullptr). Returns a heap copy of the path the caller
 * frees with safe_free, or nullptr when none is readable. */
[[nodiscard]] char *geist_find_aux_file(const char        *env_name,
                                        const char        *dir,
                                        const char        *basename,
                                        const char *const *fallbacks);

#endif /* GEIST_AUX_FILE_H */
