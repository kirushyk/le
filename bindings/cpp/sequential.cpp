#include "sequential.hpp"
#include <le/le.h>

using namespace le;

Sequential::Sequential()
{
    setCModel(LE_MODEL(le_sequential_new()));
}
