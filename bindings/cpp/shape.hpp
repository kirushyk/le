#include <cstdint>
#include <memory>
#include <le/le.h>

namespace le
{
    
class Shape
{
public:
    Shape(unsigned numDimensions);
    ~Shape();

    LeShape *c_shape();
    std::uint32_t & operator [](std::size_t index);

private:
    struct Private;
    std::shared_ptr<Private> priv;
};

}
