#include "svm.hpp"

using namespace le;

LeSVMTrainingOptions c_training_options(le::SVM::TrainingOptions options)
{
    LeSVMTrainingOptions c_options;
    c_options.kernel = (LeKernel)options.kernel;
    c_options.c = options.c;
    return c_options;
}

SVM::SVM()
{
    setCModel(LE_MODEL(le_svm_new()));
}

void SVM::train(const Tensor &x_train, const Tensor &y_train, TrainingOptions options)
{
    le_svm_train(LE_SVM(c_model()),
        x_train.c_tensor(), y_train.c_tensor(),
        c_training_options(options));
}
