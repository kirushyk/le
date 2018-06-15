#include "base.h"
#include <le/le.h>

BaseClass base_class;

void
base_construct(Base *base)
{
    ((LeObject *)base)->klass = (LeClass *)&base_class;
}

char
base_polymorphic(Base *base)
{
    return ((BaseClass *)((LeObject *)base)->klass)->base_polymorphic(base);
}
