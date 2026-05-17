# Diffusion Project

This project trains a small DDPM-style diffusion model on CelebA faces.

The Colab notebook is meant to show the result directly:

1. install the project
2. load CelebA
3. train for 10 epochs and print the loss after each epoch
4. display 4 generated samples at timesteps `0, 200, 400, 600, 800, 1000`
5. display a looping GIF that goes from noise to a generated image

The notebook does not save result images locally. It only displays them in Colab.
