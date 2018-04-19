#ifndef OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP
#define OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP

#include <vector>
#include <functional>

namespace methods {
    class ConstGradientDescent {
    public:
        using Function = std::function<float(std::vector<float>)>;

        ConstGradientDescent(const std::vector<float> &values, const Function &function);

        std::vector<float> optimize();

        virtual ~ConstGradientDescent();

    private:
        std::vector<float> m_values;
        Function m_function;
        float m_delta = 0.0001;
        float m_lambda = 0.01;
        float m_precision = 0.00001;
    };

    void subtract(std::vector<float> vec1, std::vector<float> vec2, std::vector<float> &result);
}

#endif //OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP
