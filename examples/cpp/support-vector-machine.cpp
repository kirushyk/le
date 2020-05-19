#include <iostream>
#include <le.hpp>

int main(int argc, const char *argv[])
{
    le::Tensor x(le::Type::FLOAT32, 2, 2, 4,
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    );

    le::Tensor y(le::Type::FLOAT32, 2, 1, 4,
        -1.0f, -1.0f, 1.0f, 1.0f
    );
    
    std::cout << "Train set: " << std::endl;
    std::cout << "x =" << std::endl << x << std::endl;
    std::cout << "y =" << std::endl << y << std::endl;

    le::SVM svm;
    le::SVM::TrainingOptions options;
    options.kernel = le::Kernel::LINEAR;
    options.c = 1.0f;
    svm.train(x, y, options);

    le::Tensor h = svm.predict(x);
    std::cout << "Predicted value =" << std::endl << h << std::endl;
    
    return 0;
}

