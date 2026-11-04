/* A vtbl without a required slot does not compile: a missing argument. */
#include "backend_fixture.h"

#ifdef WIRING_OK
const struct geist_backend_vtbl fixture_vtbl = FIXTURE_VTBL(FIXTURE_VTBL_SLOTS);
#else
// expect: too few arguments|requires [0-9]+ arguments|requires at least
const struct geist_backend_vtbl fixture_vtbl = GEIST_BACKEND_VTBL(fixture_destroy,
                                                                  geist_host_buffer_create,
                                                                  geist_host_buffer_destroy,
                                                                  geist_host_buffer_create_aliased,
                                                                  geist_host_buffer_upload,
                                                                  geist_host_buffer_download,
                                                                  geist_host_buffer_map);
#endif
