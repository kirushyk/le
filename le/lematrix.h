/**
  *
  */

#ifndef __LEMATRIX_H_
#define __LEMATRIX_H_

typedef struct LeMatrix LeMatrix;

LeMatrix * le_matrix_new                (void);

LeMatrix * le_matrix_new_copy           (LeMatrix     *another);

LeMatrix * le_matrix_new_from_data      (unsigned      height,
                                         unsigned      width,
                                         const float  *data);

unsigned   le_matrix_get_width          (LeMatrix     *matrix);

unsigned   le_matrix_get_height         (LeMatrix     *matrix);

float      le_matrix_at                 (LeMatrix     *matrix,
                                         unsigned      y,
                                         unsigned      x);

LeMatrix * le_matrix_new_identity       (unsigned      size);

LeMatrix * le_matrix_new_zeros          (unsigned      height,
                                         unsigned      width);

LeMatrix * le_matrix_new_rand           (unsigned      height,
                                         unsigned      width);

LeMatrix * le_matrix_new_transpose      (LeMatrix     *a);

LeMatrix * le_matrix_new_product        (LeMatrix     *a,
                                         LeMatrix     *b);

void       le_matrix_subtract           (LeMatrix     *a,
                                         LeMatrix     *b);

void       le_matrix_multiply_by_scalar (LeMatrix     *a,
                                         float         b);

void       le_matrix_add_scalar         (LeMatrix     *a,
                                         float         b);

float      le_matrix_sum                (LeMatrix     *matrix);

void       le_matrix_apply_sigmoid      (LeMatrix     *matrix);

void       le_matrix_free               (LeMatrix     *matrix);

/** @note: Temporary */
#include <stdio.h>

void       le_matrix_print              (LeMatrix     *matrix,
                                         FILE         *stream);

#endif
