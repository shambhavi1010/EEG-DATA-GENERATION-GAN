# EEG-DATA-GENERATION-GAN
<div style="border-radius:12px; padding: 20px; background-color: #d5d5ed; font-size:120%; text-align:center">
# EEG-DATA-SYNTHESIS-USING-GANS

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Workflow](#workflow)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Used](#model-used)
6. [Training](#training)
7. [Results](#results)  
8. [Future Work](#future-work)
9. [Conclusion](#conclusion)
# Introduction
Electroencephalography (EEG) serves as a fundamental tool in the diagnosis and monitoring of epilepsy, a neurological disorder characterized by recurrent seizures. EEG recordings provide valuable insights into the dynamic electrical activity of the brain, aiding clinicians in identifying epileptiform patterns and guiding treatment decisions. However, the analysis of EEG data is often challenged by limitations such as data scarcity, artifacts, and variability across patients.
In recent years, deep learning techniques, particularly Generative Adversarial Networks (GANs), have emerged as promising approaches for enhancing the quality and utility of EEG data. GANs offer a powerful framework for generating synthetic EEG samples that closely resemble real-world recordings, enabling data augmentation and artifact removal while preserving the underlying physiological characteristics.

# Installation
To run the code in this project, you will need the following dependencies installed:

1.Python: Version 3.6 or higher
2.TensorFlow: Version 2.0 or higher
3.NumPy: For numerical computations
4.Matplotlib: For data visualization
5.Pandas: For data manipulation and preprocessing
# Workflow 
![Flowchart Building (1)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/109196162/427f3e8b-7df1-4565-8f3b-a04d9143a14f)



# Dataset Preparation

The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds. The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.
We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.

All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Our motivation for creating this version of the data was to simplify access to the data via the creation of a .csv version of it. Although there are 5 classes most authors have done binary classification, namely class 1 (Epileptic seizure) against the rest.

# Model Used
Generative Adversarial Network (GAN) aimed at generating EEG data related to epileptic and non-epileptic seizures. It begins by constructing a discriminator network using TensorFlow and Keras, designed to classify inputs as either real or generated data. This discriminator consists of a dense network with a sigmoid output layer, appropriate for binary classification, and is compiled with a binary cross-entropy loss function and SGD optimizer. A simplistic generator model is also defined, which will transform latent space vectors into data points meant to mimic the real EEG data's structure.

A GAN class is then defined, which encapsulates both the generator and discriminator. This class includes a custom training step that involves generating fake data, combining it with real data, and then training the discriminator on this mixture. The generator is subsequently trained to fool the discriminator using misleading labels. The GAN is compiled with Adam optimizers for both the generator and discriminator, and a binary cross-entropy loss function configured to align with the sigmoid activation in the discriminator. Finally, the GAN is trained using EEG data for a specified number of epochs. This setup aims to allow the generator to learn to produce new EEG-like data that mimics the distribution of the real EEG data used in training, potentially aiding in tasks like data augmentation for training other models or studying seizure-related EEG patterns.




Wave GAN: WaveGAN typically consists of a generator and a discriminator, similar to other GAN architectures. However, it incorporates specific design choices tailored for waveform generation, such as using transposed convolutions in the generator to upsample the input noise into waveform data.
Generator:
In the context of EEG data synthesis, the generator in WaveGAN would take random noise as input and generate synthetic EEG signals. The generator architecture may include transposed convolutional layers, often combined with upsampling techniques like nearest-neighbor interpolation or deconvolution, to transform the input noise into time-series data resembling EEG waveforms.
Discriminator:
The discriminator in WaveGAN is responsible for distinguishing between real EEG data and synthetic EEG data generated by the generator. It typically consists of convolutional layers followed by fully connected layers, designed to process waveform data and make a binary classification (real vs. synthetic).

# Training
The training process for the WaveGAN model is centered around the use of real EEG data that must be preprocessed and shaped correctly before being fed into the model. The steps described below outline the training process implemented in the provided code.

### Data Preparation
Before training can begin, your EEG data needs to be formatted correctly:
- **Reshaping**: EEG data should be reshaped into the required format `(batch_size, 1024, 1)`. This ensures compatibility with the input shape expected by the discriminator model.

### Training Steps
The training process involves several steps that are executed in each epoch:

1. **Latent Space Sampling**:
   - For each batch, generate a set of random latent vectors. These vectors serve as the input for the generator to produce synthetic EEG data.

2. **Generating EEG Data**:
   - The generator takes the random latent vectors and produces EEG data that attempts to mimic real EEG recordings.

3. **Discriminator Evaluation**:
   - The discriminator evaluates both real and generated EEG data. It outputs a probability distribution over the authenticity of the data, i.e., real or fake.

4. **Loss Calculation**:
   - Losses are calculated separately for both the generator and discriminator:
     - **Generator Loss**: Encourages the generator to produce data that the discriminator will classify as real.
     - **Discriminator Loss**: Composed of two parts—loss from real EEG data (should be classified as real) and loss from generated data (should be classified as fake). The total discriminator loss is the sum of these two parts.

5. **Backpropagation and Optimization**:
   - Using the losses calculated, gradients are computed for both the generator and discriminator.
   - The gradients are applied to update the weights of both models using their respective optimizers. This optimization step is crucial for learning the distribution of the real EEG data.

### Epochs and Batch Processing
- The training process is iterated over a specified number of epochs. In each epoch, all steps from latent space sampling to optimization are repeated until all batches of the EEG data are processed.

### Monitoring and Adjustments
- **Monitoring**: It's important to monitor the generator and discriminator losses to understand the training dynamics. Adjustments might be necessary if the model shows signs of instability (e.g., mode collapse, where the generator produces limited variety of outputs).
- **Adjustments**: Parameters such as the learning rate, batch size, and number of training epochs may be tuned based on the observed performance and training dynamics.

This detailed training process helps in guiding the development and optimization of the WaveGAN model tailored for EEG data synthesis, aiming to produce high-quality, realistic EEG-like signals.

# Results

Comparison Of Real & Generated Epileptic Data
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/7db704e6-f3bb-4be1-b4f8-861d6717bfa8)

Comparison Of Real & Generated Non- Epileptic Data
![download (1) (2)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/6a99207d-e75e-4949-a0c7-38e782d8f778)




No. Of Epochs: 2000
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/37171f36-179e-41d2-b143-c8ba73ae4e36)
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/86b44362-68cf-4f07-bcec-9bc2781692dc)

WaveGAN Result: 
No. Of epochs = 5000

![download (2)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/109196162/02dc7215-9cd1-456d-b2b3-93fab61160b8)



# Future Work

Improved Model Architectures: Experiment with more advanced GAN architectures or modifications of existing architectures tailored specifically for EEG data synthesis. Explore techniques such as Wasserstein GANs (WGANs), Progressive GANs, or attention mechanisms to enhance the quality and diversity of synthetic EEG data.
Multi-Modal Data Synthesis: Incorporate additional modalities, such as simultaneous EEG and fMRI data, to create multi-modal synthetic datasets. This could provide richer representations of brain activity and facilitate more comprehensive studies on epilepsy and related disorders. 

# Conclusion

The implementation of WaveGAN for synthesizing EEG data has demonstrated significant potential in generating realistic EEG signals. This project not only serves as a proof of concept but also opens up numerous possibilities for further research and development in the field of EEG analysis and synthesis.

### Achievements
- **Successful Implementation**: The WaveGAN model was successfully adapted and implemented for the specific task of EEG data synthesis, showing good performance in generating believable EEG-like signals.
- **Advancement in EEG Research**: By providing a method to generate synthetic EEG data, this project contributes to EEG research, offering a tool that can be used for various applications such as data augmentation, testing of EEG processing algorithms, and training of machine learning models without the need for extensive real data.

