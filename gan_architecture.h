#ifndef GAN_ARCHITECTURE_H
#define GAN_ARCHITECTURE_H

#include <vector>
#include <string>

// Generator class
class Generator {
public:
    Generator(int inputDim, int outputDim);
    std::vector<float> forward(const std::vector<float>& input);
    void initializeWeights();  // Initialize weights for the generator
    int inputDim;
    std::vector<float> weights;

private:
    int outputDim;
};

// Discriminator class
class Discriminator {
public:
    Discriminator(int inputDim);
    float forward(const std::vector<float>& input);
    void initializeWeights();  // Initialize weights for the discriminator
    std::vector<float> weights;

private:
    int inputDim;
}; 

// GAN class for training and evaluation
class GAN {
public:
    GAN(int noiseDim, int imageDim);

    void train(int epochs, int batchSize);
    void generateImage(const std::vector<float>& noise, std::vector<float>& generatedImage);
    void saveImage(const std::vector<float>& image, int width, int height, const std::string& filename);

private:
    Generator generator;         // Generator model
    Discriminator discriminator; // Discriminator model
};

#endif // GAN_ARCHITECTURE_H
