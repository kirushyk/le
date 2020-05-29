#include <iostream>
#include <le.hpp>

int main(int argc, const char *argv[])
{
    le::Tensor x(le::Type::FLOAT32, 2, 2, 4,
        1.0, 2.0, 1.0, 2.0,
        2.0, 2.0, 1.0, 1.0
    );

    le::Tensor y(le::Type::FLOAT32, 2, 1, 4,
        0.0, 1.0, 1.0, 0.0
    );
    
    std::cout << "Train set: " << std::endl;
    std::cout << "x =" << std::endl << x << std::endl;
    std::cout << "y =" << std::endl << y << std::endl;

    std::cout << "Creating Neural Network Structure" << std::endl;
    le::Sequential nn;
    nn.add(le::DenseLayer("D1", 2, 2));
    nn.add(le::ActivationLayer("A1", le::Activation::SIGMOID));
    nn.add(le::DenseLayer("D2", 2, 1));
    nn.add(le::ActivationLayer("A2", le::Activation::SIGMOID));
    nn.setLoss(le::Loss::LOGISTIC);

    std::cout << "Training Neural Network" << std::endl;
    le::BGD optimizer(nn, x, y, 3.f);
    for (unsigned i = 0; i <= 1000; i++)
    {
        optimizer.step();

        if ((i % 100) == 0)
        {
            std::cout << "Iteration " << i << std::endl;
            le::Tensor h = nn.predict(x);
            std::cout << "Training Error = " << le_logistic_loss(h.c_tensor(), y.c_tensor()) << std::endl;
        }
    }

    le::Tensor h = nn.predict(x);
    std::cout << "Predicted value =" << std::endl << h << std::endl;
    
    return 0;
}
