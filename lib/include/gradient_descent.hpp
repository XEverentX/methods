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
                             float precision = 0.0001,
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

    class FractionalGradientDescent : public IGradientDescent {
    public:
        FractionalGradientDescent(std::vector<float> values,
                                  methods::ConstGradientDescent::Function function,
                                  float precision = 0.0001,
                                  float delta = 0.1,
                                  float lambda = 0.01,
                                  float sigma = 0.95,
                                  float epsilon = 0.1);

        std::vector<float> optimize() override;

        const std::vector<float> &getValues() const;

        void setValues(const std::vector<float> &m_values);

        const Function &getFunction() const;

        void setFunction(const Function &m_function);

        float getPrecision() const;

        void setPrecision(float m_precision);

        float getDelta() const;

        void setDelta(float m_delta);

        float getLambda() const;

        void setLambda(float m_lambda);

        float getSigma() const;

        void setSigma(float m_sigma);

        float getEpsilon() const;

        void setEpsilon(float m_epsilon);

        ~FractionalGradientDescent() override;

    private:
        std::vector<float> m_values;
        Function m_function;
        float m_precision;
        float m_delta;
        float m_lambda;
        float m_sigma;
        float m_epsilon;
    };

    void subtract(std::vector<float> vec1, std::vector<float> vec2, std::vector<float> &result);
}

#endif //OPTIMIZATION_TRY_GRADIENT_DESCENT_HPP
