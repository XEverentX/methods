#ifndef OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP
#define OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP

#include <vector>
#include <functional>

namespace methods {
    class IGradientDescent {
    public:
        using Function = std::function<float(std::vector<float>)>;

        virtual std::vector<float> optimize() = 0;

        virtual ~IGradientDescent() = default;
    };

    class ConstGradientDescent : public IGradientDescent {
    public:

        ConstGradientDescent(std::vector<float> values,
                             methods::ConstGradientDescent::Function function,
                             float precision = 0.01,
                             float delta = 0.1,
                             float lambda = 0.01);

        std::vector<float> optimize() override;

        float getM_delta() const;

        void setM_delta(float m_delta);

        float getM_lambda() const;

        void setM_lambda(float m_lambda);

        float getM_precision() const;

        void setM_precision(float m_precision);

        ~ConstGradientDescent() override;

    private:
        std::vector<float> m_values;
        Function m_function;
        float m_precision;
        float m_delta;
        float m_lambda;
    };

    void subtract(std::vector<float> vec1, std::vector<float> vec2, std::vector<float> &result);
}

#endif //OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP
