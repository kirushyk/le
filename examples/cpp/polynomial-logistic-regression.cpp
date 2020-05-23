#include <iostream>
#include <le.hpp>

int main(int argc, char *argv[])
{
    le::Tensor x(le::Type::FLOAT32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        4.0, 3.0, 2.0, 1.0
    );

    le::Tensor y(le::Type::FLOAT32, 2, 1, 4,
        0.0, 0.0, 1.0, 1.0
    );

    std::cout << "Train set: " << std::endl;
    std::cout << "x =" << std::endl << x << std::endl;
    std::cout << "y =" << std::endl << y << std::endl;

    le::LogisticClassifier lc;
    le::LogisticClassifier::TrainingOptions options;
    options.maxIterations = 100;
    options.learningRate = 1.0f;
    options.polynomiaDegree = 1;
    options.regularization = le::Regularization::NONE;
    options.lambda = 0.0f;
    lc.train(x, y, options);

    le::Tensor h = lc.predict(x);
    std::cout << "Predicted value =" << std::endl << h << std::endl;

    return 0;
}
