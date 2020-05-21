#include <cstdint>
#include <memory>
#include <le/le.h>

namespace le
{
    
class Shape
{
public:
    Shape(unsigned numDimensions, std::uint32_t *sizes);
    ~Shape();

    LeShape *c_shape();

private:
    struct Private;
    std::shared_ptr<Private> priv;
};

}
