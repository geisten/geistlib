/* Included only in the scratch engine used for Qwen case 47. */
#include "qwen-m1-rounding-io.h"
#include <stdio.h>

[[nodiscard]] static bool q1_load_buffer(uint32_t                first,
                                         uint32_t                last,
                                         const char             *path,
                                         struct metal_buffer_h **out,
                                         struct metal_state     *st) {
    if (path == nullptr)
        return false;
    FILE *file = fopen(path, "rb");
    if (file == nullptr)
        return false;
    const long   length     = fseek(file, 0, SEEK_END) == 0 ? ftell(file) : -1;
    const size_t max_values = (size_t) (last - first + 1u);
    const size_t max_blocks = (max_values + Q1_BLOCK_SIZE - 1u) / Q1_BLOCK_SIZE;
    const size_t max_bytes =
            sizeof(uint32_t) * (Q1_HEADER_WORDS + max_blocks * Q1_BLOCK_WORDS + max_values);
    bool ok = length > 0 && (size_t) length <= max_bytes &&
              (size_t) length % sizeof(uint32_t) == 0 && fseek(file, 0, SEEK_SET) == 0;
    if (ok) {
        *out                = metal_msg_send_id_size_uint(st,
                                                          st->device,
                                                          "newBufferWithLength:options:",
                                                          (size_t) length,
                                                          METAL_RESOURCE_STORAGE_MODE_SHARED);
        uint32_t    *mapped = *out == nullptr ? nullptr : metal_msg_send_id0(st, *out, "contents");
        const size_t count  = (size_t) length / sizeof(*mapped);
        ok = mapped != nullptr && fread(mapped, sizeof(*mapped), count, file) == count &&
             fgetc(file) == EOF && !ferror(file) && q1_table_valid(count, mapped) &&
             mapped[2] == first && mapped[3] == last;
    }
    fclose(file);
    return ok;
}
