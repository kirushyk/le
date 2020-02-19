#include "letensorlist.h"
#include <stdio.h>
#include <stdint.h>

void
le_tensorlist_save(LeList *tensors, const char *filename)
{
    FILE *fout = fopen(filename, "wb");
    uint8_t version = 0;
    fwrite(&version, 1, 1, fout);
    fclose(fout);
}

LeList *
le_tensorlist_load(const char *filename)
{
    return NULL;
}
