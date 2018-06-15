#include "a.h"
#include "base.h"
#include <stdlib.h>

struct A
{
    Base parent;
};

typedef struct AClass
{
    BaseClass parent;
} AClass;

AClass *a_class = NULL;

char
a_polymorphic(A *a)
{
    return 'a';
}

void
a_class_init(AClass *a_class)
{
    a_class->parent.base_polymorphic = (char(*)(Base *base))a_polymorphic;
}

A *
a_alloc()
{
    A *a = malloc(sizeof(A));
    return a;
}

void
a_construct(A *a)
{
    base_construct((Base *)a);
    if (a_class == NULL)
    {
        a_class = malloc(sizeof(AClass));
        a_class_init(a_class);
    }
    ((LeObject *)a)->klass = (LeClass *)a_class;
    a->parent.value = 13;
}

A *
a_new(void)
{
    A *a = a_alloc();
    a_construct(a);
    return a;
}

void
a_free(A *a)
{
    free(a);
}
