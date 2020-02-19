#include "letensorlist.h"
#include <stdio.h>
#include <stdint.h>

void
le_tensorlist_save(LeList *tensors, const char *filename)
{
    FILE *fout = fopen(filename, "wb");
    if (fout)
    {
        uint8_t version = 1;
        fwrite(&version, 1, 1, fout);
        fclose(fout);
    }
}

LeList *
le_tensorlist_load(const char *filename)
{
    LeList *list = NULL;
    FILE *fin = fopen(filename, "rb");
    if (fin)
    {
        uint8_t version = 0;
        fread(&version, 1, 1, fin);
        fclose(fin);
    }
    return list;
}
