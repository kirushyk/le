#include <stdio.h>
#include <le/le.h>
#include "matrices/letests-matrices.h"

int
main(int argc, const char *argv[])
{
    if (le_test_matrices())
        return 1;
    
    printf("All tests passed.\n");
    return 0;
}
