#include "b.h"
#include "base.h"
#include <stdlib.h>

/* Subclass B */

typedef struct B
{
    Base parent;
} B;

char
b_polymorphic(B *b)
{
    return 'b';
}

typedef struct BClass
{
    BaseClass parent;
} BClass;

BClass *b_class = NULL;

void
b_class_init(BClass *b_class)
{
    b_class->parent.base_polymorphic = (char(*)(Base *base))b_polymorphic;
}

B *
b_alloc()
{
    B *b = malloc(sizeof(B));
    return b;
}

void
b_construct(B *b)
{
    base_construct((Base *)b);
    if (b_class == NULL)
    {
        b_class = malloc(sizeof(BClass));
        b_class_init(b_class);
    }
    ((LeObject *)b)->klass = (LeClass *)b_class;
    b->parent.value = 42;
}

B *
b_new(void)
{
    B *b = b_alloc();
    b_construct(b);
    return b;
}

void
b_free(B *b)
{
    free(b);
}
