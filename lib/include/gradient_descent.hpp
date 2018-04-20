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

        float getDelta() const;

        void setDelta(float m_delta);

        float getLambda() const;

        void setLambda(float m_lambda);

        float getPrecision() const;

        void setPrecision(float m_precision);

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
