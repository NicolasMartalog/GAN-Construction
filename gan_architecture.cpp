#include "gan_architecture.h"
#include "matrix_mult.h"
#include <iostream>
#include <random> 
#include "adam_optimizer.h"
#include <opencv2/opencv.hpp>
#include <cmath>

float binaryCrossEntropy(float predicted, float target) {
    // Prevent log(0) issues
    predicted = std::max(std::min(predicted, 0.999999f), 0.000001f);
    return - (target * log(predicted) + (1 - target) * log(1 - predicted));
} 

Generator::Generator(int inputDim, int outputDim)
    : inputDim(inputDim), outputDim(outputDim) {
    initializeWeights();
}

void Generator::initializeWeights() {
    // Xavier/Glorot initialization
    float stddev = std::sqrt(2.0f / (inputDim + outputDim));
    
    weights.resize(inputDim * outputDim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, stddev);
    
    for (auto& w : weights) {
        w = d(gen);
    }
}

std::vector<float> Generator::forward(const std::vector<float>& input) {
    if (input.size() != inputDim) {
        throw std::invalid_argument("Input size does not match generator input dimension");
    }
    
    std::vector<float> output(outputDim, 0.0f);
    matMul(input.data(), weights.data(), output.data(), 1, inputDim, outputDim);
    
    // Apply tanh activation to constrain output
    for (auto& val : output) {
        val = std::tanh(val);
    }
    
    return output;
}

Discriminator::Discriminator(int inputDim)
    : inputDim(inputDim) {
    initializeWeights();
}

void Discriminator::initializeWeights() {
    float stddev = std::sqrt(2.0f / inputDim);
    
    weights.resize(inputDim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, stddev);
    
    for (auto& w : weights) {
        w = d(gen);
    }
}

float Discriminator::forward(const std::vector<float>& input) {
    if (input.size() != inputDim) {
        throw std::invalid_argument("Input size does not match discriminator input dimension");
    }
    
    float output = 0.0f;
    for (int i = 0; i < inputDim; ++i) {
        output += input[i] * weights[i];
    }
    
    // Sigmoid activation
    return 1.0f / (1.0f + std::exp(-output));
}

GAN::GAN(int noiseDim, int imageDim)
    : generator(noiseDim, imageDim), discriminator(imageDim) {}

void GAN::train(int epochs, int batchSize) {
    // Use more explicit optimizer with tuned parameters
    AdamOptimizer generatorOptimizer(0.0002, 0.5, 0.999);
    AdamOptimizer discriminatorOptimizer(0.0002, 0.5, 0.999);

    std::vector<float> genM(generator.weights.size(), 0.0f);
    std::vector<float> genV(generator.weights.size(), 0.0f);
    std::vector<float> discM(discriminator.weights.size(), 0.0f);
    std::vector<float> discV(discriminator.weights.size(), 0.0f);

    float lambda = 0.001f;  // Reduced regularization strength

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalDiscLoss = 0.0f;
        float totalGenLoss = 0.0f;

        for (int i = 0; i < batchSize; ++i) {
            // Generate random noise
            std::vector<float> noise(generator.inputDim);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            for (auto& n : noise) {
                n = dis(gen);
            }

            // Generator forward pass
            std::vector<float> fakeImage = generator.forward(noise);

            // Clip pixel values
            for (auto& pixel : fakeImage) {
                pixel = std::max(-1.0f, std::min(1.0f, pixel));
            }

            // Discriminator forward pass
            float realOrFake = discriminator.forward(fakeImage);

            // Discriminator Loss (Binary Cross-Entropy)
            float dLossReal = -log(std::max(1e-10f, 1.0f - realOrFake));
            float dLossFake = -log(std::max(1e-10f, realOrFake));
            float dLoss = 0.5f * (dLossReal + dLossFake);

            // Discriminator Gradient
            std::vector<float> dGradients(fakeImage.size(), 0.0f);
            for (size_t i = 0; i < fakeImage.size(); ++i) {
                dGradients[i] = realOrFake - (i < fakeImage.size() / 2 ? 1.0f : 0.0f);
            }

            // Update Discriminator
            discriminatorOptimizer.update(discriminator.weights, dGradients, discM, discV, epoch + 1);

            // Generator Gradient (tries to maximize discriminator's uncertainty)
            std::vector<float> gGradients(fakeImage.size(), 0.0f);
            for (size_t i = 0; i < fakeImage.size(); ++i) {
                gGradients[i] = -dGradients[i];
            }

            // Update Generator
            generatorOptimizer.update(generator.weights, gGradients, genM, genV, epoch + 1);

            // Accumulate losses
            totalDiscLoss += dLoss;
            totalGenLoss += -dGradients[0];  // Simplified generator loss
        }

        // Print average losses
        std::cout << "Epoch " << epoch + 1 
                  << " | Avg Disc Loss: " << totalDiscLoss / batchSize 
                  << " | Avg Gen Loss: " << totalGenLoss / batchSize 
                  << std::endl;
    }
}

void GAN::generateImage(const std::vector<float>& noise, std::vector<float>& generatedImage) {
    if (noise.size() != generator.inputDim) {
        throw std::invalid_argument("Noise vector size does not match generator input dimension");
    }
    generatedImage = generator.forward(noise);
} 

void GAN::saveImage(const std::vector<float>& image, int width, int height, const std::string& filename) {
    // Create an OpenCV Mat object with 8-bit 3-channel (BGR) data
    if (image.size() != width * height * 3) {
        std::cerr << "Image size mismatch! Expected " << width * height * 3 
                  << " got " << image.size() << std::endl;
        return;
    } 

    cv::Mat img(height, width, CV_8UC3);

    // Convert from float to 0-255 range (assuming image values are between -1 and 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3; // Index in the input vector
            img.at<cv::Vec3b>(y, x)[0] = static_cast<unsigned char>((image[idx] + 1.0f) * 127.5f);     // B
            img.at<cv::Vec3b>(y, x)[1] = static_cast<unsigned char>((image[idx + 1] + 1.0f) * 127.5f); // G
            img.at<cv::Vec3b>(y, x)[2] = static_cast<unsigned char>((image[idx + 2] + 1.0f) * 127.5f); // R
        }
    }

    // Save the image using OpenCV's imwrite function
    cv::imwrite(filename, img);
}