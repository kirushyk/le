#include <stdio.h>
#include "le.h"

const double x_data[] =
{
    1.0, 1.0,
    2.0, 2.0,
    3.0, 3.0,
    4.0, 4.0,
};

const double y_data[] =
{
    0.0,
    0.0,
    1.0,
    1.0
};

int
main(int argc, const char *argv[])
{
    LeMatrix *x = le_matrix_new_from_data(4, 2, x_data);
    LeMatrix *y = le_matrix_new_from_data(4, 1, y_data);
    le_matrix_free(y);
    le_matrix_free(x);
    return 0;
}
