#include "file_map.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool io_map_ro(const char *path, int flags, struct io_map *out, const char **err) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        io_set_err(err, "open() failed");
        return false;
    }
    const char *step = "fstat() failed";
    struct stat sb;
    if (fstat(fd, &sb) == 0) {
        const size_t size = (size_t) sb.st_size;
        void        *map  = mmap(nullptr, size, PROT_READ, flags, fd, 0);
        if (map != MAP_FAILED) {
            *out = (struct io_map) {.fd = fd, .map = map, .size = size};
            return true;
        }
        step = "mmap() failed";
    }
    const int saved = errno;
    close(fd);
    errno = saved;
    io_set_err(err, step);
    return false;
}
