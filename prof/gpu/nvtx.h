#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                            0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define VARGEN( prefix, id ) prefix##id

#define RANGE_START( name, id, cid ) \
    nvtxEventAttributes_t VARGEN(eventAttrib, id) = {0,0,0,0,0,0,0,0,0,0}; \
    VARGEN(eventAttrib, id).version = NVTX_VERSION; \
    VARGEN(eventAttrib, id).size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    VARGEN(eventAttrib, id).colorType = NVTX_COLOR_ARGB; \
    VARGEN(eventAttrib, id).color = colors[(cid) % num_colors]; \
    VARGEN(eventAttrib, id).messageType = NVTX_MESSAGE_TYPE_ASCII; \
    VARGEN(eventAttrib, id).message.ascii = (name); \
    nvtxRangeId_t VARGEN(range, id) = \
        nvtxRangeStartEx(&VARGEN(eventAttrib, id))
#define RANGE_END( id ) \
    nvtxRangeEnd(VARGEN(range, id))
#else
#define RANGE_START( name, id, cid )
#define RANGE_END( id )
#endif
