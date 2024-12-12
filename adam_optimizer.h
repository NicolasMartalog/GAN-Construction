#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include <vector>

class AdamOptimizer {
public:
    AdamOptimizer(float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    void update(std::vector<float>& weights, const std::vector<float>& gradients, std::vector<float>& m, std::vector<float>& v, int t);

private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
};

#endif // ADAM_OPTIMIZER_H
