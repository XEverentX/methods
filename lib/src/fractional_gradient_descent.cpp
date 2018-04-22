#include <gradient_descent.hpp>

methods::FractionalGradientDescent::FractionalGradientDescent(std::vector<float> values,
                                                              methods::ConstGradientDescent::Function function,
                                                              const float precision,
                                                              const float delta,
                                                              const float lambda,
                                                              const float sigma,
                                                              const float epsilon)
        : m_values(std::move(values)),
          m_function(std::move(function)),
          m_precision(precision),
          m_delta(delta),
          m_lambda(lambda),
          m_sigma(sigma),
          m_epsilon(epsilon) {}

std::vector<float> methods::FractionalGradientDescent::optimize() {
    auto resultValues = std::vector<float>();
    auto currentValues = std::vector<float>(m_values);
    auto grad = std::vector<float>();

    // Calculating initial gradient
    for (float &currentValue : currentValues) {
        auto lowBorder = m_function(currentValues);
        currentValue += m_delta;
        auto highBorder = m_function(currentValues);
        currentValue -= m_delta;
        grad.push_back(((highBorder - lowBorder) / m_delta));
    }

    auto lambda = m_lambda;
    auto trialGrad = std::vector<float>(grad);
    std::for_each(trialGrad.begin(), trialGrad.end(), [this](float &value) {
        value *= m_lambda * m_epsilon;
    });
    methods::subtract(currentValues, trialGrad, resultValues);
    while (std::abs(m_function(resultValues)) >
           std::abs(std::abs(m_function(currentValues)) - std::abs(m_function(trialGrad)))) {
        lambda *= m_sigma;
        trialGrad = grad;
        std::for_each(trialGrad.begin(), trialGrad.end(), [lambda, this](float &value) {
            value *= lambda * m_epsilon;
        });
        methods::subtract(currentValues, trialGrad, resultValues);
    }
    std::for_each(grad.begin(), grad.end(), [lambda](float &value) {
        value *= lambda;
    });
    methods::subtract(currentValues, grad, resultValues);

    // Main cycle of calculations
    while (std::abs(std::abs(m_function(resultValues)) - std::abs(m_function(currentValues))) > m_precision) {
        currentValues = resultValues;
        grad.clear();
        resultValues.clear();

        for (float &currentValue : currentValues) {
            auto lowBorder = m_function(currentValues);
            currentValue += m_delta;
            auto highBorder = m_function(currentValues);
            currentValue -= m_delta;
            grad.push_back(((highBorder - lowBorder) / m_delta) * m_lambda);
        }

        lambda = m_lambda;
        trialGrad = grad;
        std::for_each(trialGrad.begin(), trialGrad.end(), [this](float &value) {
            value *= m_lambda * m_epsilon;
        });
        methods::subtract(currentValues, trialGrad, resultValues);

        while (std::abs(m_function(resultValues)) >
               std::abs(std::abs(m_function(currentValues)) - std::abs(m_function(trialGrad)))) {
            lambda *= m_sigma;
            trialGrad = grad;
            std::for_each(trialGrad.begin(), trialGrad.end(), [lambda, this](float &value) {
                value *= lambda * m_epsilon;
            });
            methods::subtract(currentValues, trialGrad, resultValues);
        }
        std::for_each(grad.begin(), grad.end(), [lambda](float &value) {
            value *= lambda;
        });
        methods::subtract(currentValues, grad, resultValues);
    }

    return resultValues;
}

const std::vector<float> &methods::FractionalGradientDescent::getValues() const {
    return m_values;
}

void methods::FractionalGradientDescent::setValues(const std::vector<float> &m_values) {
    FractionalGradientDescent::m_values = m_values;
}

const methods::IGradientDescent::Function &methods::FractionalGradientDescent::getFunction() const {
    return m_function;
}

void methods::FractionalGradientDescent::setFunction(const methods::IGradientDescent::Function &m_function) {
    FractionalGradientDescent::m_function = m_function;
}

float methods::FractionalGradientDescent::getPrecision() const {
    return m_precision;
}

void methods::FractionalGradientDescent::setPrecision(float m_precision) {
    FractionalGradientDescent::m_precision = m_precision;
}

float methods::FractionalGradientDescent::getDelta() const {
    return m_delta;
}

void methods::FractionalGradientDescent::setDelta(float m_delta) {
    FractionalGradientDescent::m_delta = m_delta;
}

float methods::FractionalGradientDescent::getLambda() const {
    return m_lambda;
}

void methods::FractionalGradientDescent::setLambda(float m_lambda) {
    FractionalGradientDescent::m_lambda = m_lambda;
}

float methods::FractionalGradientDescent::getSigma() const {
    return m_sigma;
}

void methods::FractionalGradientDescent::setSigma(float m_sigma) {
    FractionalGradientDescent::m_sigma = m_sigma;
}

float methods::FractionalGradientDescent::getEpsilon() const {
    return m_epsilon;
}

void methods::FractionalGradientDescent::setEpsilon(float m_epsilon) {
    FractionalGradientDescent::m_epsilon = m_epsilon;
}

methods::FractionalGradientDescent::~FractionalGradientDescent() = default;
