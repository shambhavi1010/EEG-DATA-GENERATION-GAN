# Comparision and evaluation of Various GAN architectures for Synthetic EEG Data

  
We used the Epileptic Seizure Recognition csv to train our models

# Architecture
Generator And Discriminator
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/109196162/384f0c78-1ecc-4d26-bac1-b26d736b9739)

#  Code Structure

### `build_generator(latent_dim=100)`
This function constructs the generator part of the WaveGAN model. It creates a sequential Keras model which:
- Starts with a dense layer that does not use bias and reshapes the input to prepare it for transposed convolution.
- Uses several `BatchNormalization` and `LeakyReLU` layers for stabilization and non-linearity.
- Includes multiple `Conv1DTranspose` layers to upscale the input progressively to the desired output shape, which matches the dimensions required for EEG data synthesis.

### `build_discriminator()`
This function builds the discriminator part of the WaveGAN model. It also uses a sequential Keras model which:
- Begins with a `Conv1D` layer to perform the initial downsampling.
- Applies `LeakyReLU` activation for non-linearity and `Dropout` for regularization.
- Continues downsampling using additional `Conv1D` layers followed by `LeakyReLU` and `Dropout`.
- Flattens the output and uses a dense layer with sigmoid activation to classify the inputs as real or generated.

### `WaveGAN`
This class encapsulates the entire WaveGAN model, combining both the generator and discriminator:
- **`__init__(self, discriminator, generator)`**: Initializes the model with the discriminator and generator.
- **`compile(self, d_optimizer, g_optimizer, loss_fn)`**: Prepares the model for training by setting the optimizers and the loss function.
- **`train_step(self, real_data)`**: Defines the logic for a single training step which includes:
  - Generating fake data from random latent vectors.
  - Calculating the loss for both real and generated data.
  - Applying gradients to both the generator and discriminator.

### Main Execution Section
This section of the script is where the model components are instantiated, compiled, and trained:
- **Generator and Discriminator Creation**: Instances of the generator and discriminator are created using their respective functions.
- **Model Compilation**: The WaveGAN model is instantiated with the created generator and discriminator, and it is compiled with the Adam optimizer and binary cross-entropy loss.
- **Model Training**: The model is trained using real EEG data formatted to the required dimensions. Training parameters such as the number of epochs are set in this section.

# WaveGan Results 
![download (3)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/109196162/f7968515-ee04-431e-b03e-fdc57b9351ea)

