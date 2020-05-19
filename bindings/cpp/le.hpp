#include <memory>

namespace le
{

class Tensor
{
public:
    Tensor();
    ~Tensor();

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
