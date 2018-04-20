#include <iostream>
#include <gradient_descent.hpp>

float test_function(std::vector<float> values);

int main() {
    auto initialValues = std::vector<float>{-10.0f, 10.0f, 5.25f};
    auto method = methods::ConstGradientDescent(initialValues, test_function, 1);
    auto results = method.optimize();
    std::cout << "Function result before: " << test_function(initialValues) << std::endl;
    for (float &value : results) {
        std::cout << value << std::endl;
    }
    std::cout << "Function result after: " << test_function(results) << std::endl;
    return 0;
}

float test_function(std::vector<float> values) {
    return values[0] * values[0] + values[1] * values[1] - values[2] * values[2];
}