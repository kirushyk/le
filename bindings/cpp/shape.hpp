#include <cstdint>
#include <memory>
#include <le/le.h>

namespace le
{
    
class Shape
{
public:
    Shape(unsigned numDimensions);
    Shape(const Shape &another);
    ~Shape();

    LeShape *c_shape();
    std::size_t & operator [](std::size_t index);

private:
    struct Private;
    std::shared_ptr<Private> priv;
};

}
