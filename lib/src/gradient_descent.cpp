#include <gradient_descent.hpp>
#include <iostream>

methods::ConstGradientDescent::ConstGradientDescent(const std::vector<float> &values,
                                                    const methods::ConstGradientDescent::Function &function)
        : m_values(values), m_function(function) {}

methods::ConstGradientDescent::~ConstGradientDescent() {}

std::vector<float> methods::ConstGradientDescent::optimize() {
    auto resultValues = std::vector<float>();
    auto currentValues = std::vector<float>(m_values);
    auto grad = std::vector<float>();
    for (float &currentValue : currentValues) {
        auto lowBorder = m_function(currentValues);
        currentValue += m_delta;
        auto highBorder = m_function(currentValues);
        currentValue -= m_delta;
        grad.push_back(((highBorder - lowBorder) / m_delta) * m_lambda);
    }
    methods::subtract(currentValues, grad, resultValues);

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

void methods::subtract(std::vector<float> vec1, std::vector<float> vec2, std::vector<float> &result) {
    for (int i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] - vec2[i]);
    }
}