#include <gradient_descent.hpp>
#include <iostream>

methods::ConstGradientDescent::ConstGradientDescent(std::vector<float> values,
                                                    methods::ConstGradientDescent::Function function,
                                                    const float precision,
                                                    const float delta,
                                                    const float lambda)
        : m_values(std::move(values)),
          m_function(std::move(function)),
          m_precision(precision),
          m_delta(delta),
          m_lambda(lambda) {}

methods::ConstGradientDescent::~ConstGradientDescent() = default;

std::vector<float> methods::ConstGradientDescent::optimize() {
    auto resultValues = std::vector<float>();
    auto currentValues = std::vector<float>(m_values);
    auto grad = std::vector<float>();

    // Calculating initial gradient
    for (float &currentValue : currentValues) {
        auto lowBorder = m_function(currentValues);
        currentValue += m_delta;
        auto highBorder = m_function(currentValues);
        currentValue -= m_delta;
        grad.push_back(((highBorder - lowBorder) / m_delta) * m_lambda);
    }
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
        methods::subtract(currentValues, grad, resultValues);
    }
    return resultValues;
}

float methods::ConstGradientDescent::getDelta() const {
    return m_delta;
}

void methods::ConstGradientDescent::setDelta(float m_delta) {
    ConstGradientDescent::m_delta = m_delta;
}

float methods::ConstGradientDescent::getLambda() const {
    return m_lambda;
}

void methods::ConstGradientDescent::setLambda(float m_lambda) {
    ConstGradientDescent::m_lambda = m_lambda;
}

float methods::ConstGradientDescent::getPrecision() const {
    return m_precision;
}

void methods::ConstGradientDescent::setPrecision(float m_precision) {
    ConstGradientDescent::m_precision = m_precision;
}

void methods::subtract(std::vector<float> vec1, std::vector<float> vec2, std::vector<float> &result) {
    result.clear();
    for (int i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] - vec2[i]);
    }
}
