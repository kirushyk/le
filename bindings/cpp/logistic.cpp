#include "logistic.hpp"

using namespace le;

LeLogisticClassifierTrainingOptions c_training_options(le::LogisticClassifier::TrainingOptions options)
{
    LeLogisticClassifierTrainingOptions c_options;
    c_options.polynomia_degree = options.polynomia_degree;
    c_options.learning_rate = options.learning_rate;
    c_options.regularization = (LeRegularization)options.regularization;
    c_options.lambda = options.lambda;
    c_options.max_iterations = options.max_iterations;
    return c_options;
}

LogisticClassifier::LogisticClassifier()
{
    setCModel(LE_MODEL(le_logistic_classifier_new()));
}

void LogisticClassifier::train(const Tensor &x_train, const Tensor &y_train, TrainingOptions options)
{
    le_logistic_classifier_train(LE_LOGISTIC_CLASSIFIER(c_model()),
        x_train.c_tensor(), y_train.c_tensor(),
        c_training_options(options));
}
