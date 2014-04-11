#include "nvtx.h"
#ifdef USE_NVTX
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00,
                    0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
#else
const int colors[1] = {0};
#endif
