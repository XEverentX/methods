#include <iostream>
#include <gradient_descent.hpp>

float test_function(std::vector<float> values);

int main() {
    auto initialValues = std::vector<float>{-10.0f, 10.0f, 5.25f};
    auto constMethod = methods::ConstGradientDescent(initialValues, test_function);
    auto fractionalMethod = methods::FractionalGradientDescent(initialValues, test_function);
    auto constResults = constMethod.optimize();
    auto fractionalResults = fractionalMethod.optimize();

    std::cout << "Function result before const gradient descent: " << test_function(initialValues) << std::endl;
    for (float &value : constResults) {
        std::cout << value << std::endl;
    }
    std::cout << "Function result after const gradient descent: " << test_function(constResults) << std::endl;

    std::cout << "Function result before fractional gradient descent: " << test_function(initialValues) << std::endl;
    for (float &value : fractionalResults) {
        std::cout << value << std::endl;
    }
    std::cout << "Function result after fractional gradient descent: " << test_function(fractionalResults) << std::endl;

    return 0;
}

float test_function(std::vector<float> values) {
    return values[0] * values[0] + values[1] * values[1] - values[2] * values[2];
}