#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <opencv2/opencv.hpp> // Include OpenCV for image saving
#include "gan_architecture.h"

// Simulate data loading
std::vector<float> loadDataset(const std::string& datasetName) {
    int width = 64; // Assuming 64x64 RGB images
    int height = 64;
    std::vector<float> data(width * height * 3); // RGB channels

    for (auto& d : data) {
        d = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Values in [-1, 1]
    }

    return data;
}

int main() {
    // Initialize dataset
    auto celebAData = loadDataset("CelebA");

    // GAN parameters
    const int noiseDim = 100;
    const int imgDim = 64 * 64 * 3; // 64x64 RGB image

    // Create GAN
    GAN gan(noiseDim, imgDim);

    // Training parameters
    const int epochs = 200;
    const int batchSize = 64;

    // Train the GAN
    gan.train(epochs, batchSize); 

    // After training
    std::vector<float> noise(noiseDim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(-1.0f, 1.0f);

    for (auto& n : noise) {
        n = dist(gen);
    }

    std::vector<float> generatedImage;
    gan.generateImage(noise, generatedImage);

    // Uncomment and modify as needed
    gan.saveImage(generatedImage, 64, 64, "final_generated_image.png");

    // Generate and save an image after training
    //std::vector<float> noise(noiseDim);
    //for (auto& n : noise) n = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    //std::vector<float> generatedImage;
    //gan.generateImage(noise, generatedImage);

    //gan.saveImage(generatedImage, 64, 64, "final_generated_image.png");

    std::cout << "Training complete! Final image saved as 'final_generated_image.png'." << std::endl;

    return 0;
}

