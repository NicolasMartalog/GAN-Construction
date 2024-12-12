#include "adam_optimizer.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

void AdamOptimizer::update(std::vector<float>& weights, const std::vector<float>& gradients, 
                            std::vector<float>& m, std::vector<float>& v, int t) {
    // Resize moment vectors if they don't match weights size
    m.resize(weights.size(), 0.0f);
    v.resize(weights.size(), 0.0f);

    // If gradients are smaller, pad with zeros
    std::vector<float> paddedGradients = gradients;
    paddedGradients.resize(weights.size(), 0.0f);

    for (size_t i = 0; i < weights.size(); ++i) {
        // Compute biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * paddedGradients[i];
        
        // Compute biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * paddedGradients[i] * paddedGradients[i];

        // Compute bias-corrected first moment estimate
        float m_hat = m[i] / (1 - std::pow(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[i] / (1 - std::pow(beta2, t));

        // Update parameters
        weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}