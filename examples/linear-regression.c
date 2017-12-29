#include <stdio.h>
#include <math.h>
#include "le.h"

const double x_data[] =
{
    1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0
};

const double y_data[] =
{
    0.0, 0.0, 1.0, 1.0
};

double predict(LeMatrix *w, double b, LeMatrix *x)
{
    LeMatrix *wt = le_matrix_new_transpose(w);
    LeMatrix *pa = le_matrix_new_product(wt, x);
    double a = tanh(le_matrix_at(pa, 0, 0) + b);
    le_matrix_free(pa);
    le_matrix_free(wt);
    return a;
}

int
main(int argc, const char *argv[])
{
    LeMatrix *x = le_matrix_new_from_data(2, 4, x_data);
    LeMatrix *y = le_matrix_new_from_data(1, 4, y_data);
    LeMatrix *w = le_matrix_new_rand(2, 1);
    double b = 0.0;
    printf("Preficted value: %lf\n", predict(w, b, x));
    le_matrix_free(w);
    le_matrix_free(y);
    le_matrix_free(x);
    return 0;
}
