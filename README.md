# DCGAN
DCGAN trained on pokemon images

For general GAN theory please refer to *Theory.md*. 

## Architecture:

1. Resized images used to train discriminator to 64x64 px rgb images to ensure model parameters fit in gpu vram. 
2. Scale pixel values to {-1, 1} for input images.
3. An Adam optimizer is used with learning rate of 0.0002. The beta-1 value is set to 0.5 to slow the training and help toward convergence without causing instability. 
4. Images are generated in a batch size of 16 per epoch.
5. The model is trained for 10000 epochs and images are saved after every 100 epochs for a total of 10 batches. These are then used to create a gif to visualize training progress over 10000 epochs. 

### Generator:

1. For the generator a sequential keras model is used with leakyReLU activation to solve the dying reLU problem for the individual layers. The value of alpha is kept at the default 0.01.
2. We use random noise of dimension (100, ) as input for the generator to train.
3. Weights have been initialized using the Xavier GLOROT scheme. 
4. We use BatchNormalization to ensure that the training is stabilized. 
5. A tanh activation function is used in the final output layer of the generator to result in outputs belonging to the range {-1, 1}.
6. Binary cross entropy loss is used for the generator as it needs to learn how to generate real images from noise through back propagation. Therefore, the generator needs to generate images with the probabiity of 1.

### Discriminator: 

1. The loss function used is the binary cross entropy function alongside a sigmoid activation to get values in the range of {0, 1} for the loss function. It's primary objective is to classify real images as 1 and fake images that are generated by the generator as 0.
2. As such there are two separate losses we predefine, fake images get a value of 0 and real images from training get a value of 1.
3. No pooling layer is used for downsampling, instead we use strided convolutions for the network to learn it's own spatial downsampling.


## Output: 

Here we have a 4 x 4 grid of outputs from the final epoch: 

![image](https://user-images.githubusercontent.com/80246631/142175513-66397caa-1b79-4a6e-8e68-4ee7bc4d3b12.png)

We can observe a few trends in the output

- The images have a respectable amount of diversity by the end of training. All the images generated by epoch 100 (refer `image_at_epoch_0100.png`) were devoid of proper edges and were primarily of blue color.
- Some images have a lot of noise in them and as such higher level features such as the edges have been lost. 
- Comparing with results online we can see that some examples do resemble the outline of Pokemon. Furthermore, we can conclude that the lack of overall depth of the network combined with a small image dimension of 64 x 64 px made it difficult for the generator to reach a point where it could generate deeper features. 


![image](https://user-images.githubusercontent.com/80246631/142178623-e3b0b26d-877e-4ce6-b4aa-9b5cd6e587aa.png) 

![image](https://user-images.githubusercontent.com/80246631/142178881-57d43044-a968-47d5-be98-12877837283f.png)



## Evaluation: 

It is clear from observing the output that despite training for 10000 epochs this model is not too capable of generating images of Pokemon. By virtue of their architecture GANs are difficult to train. However, the output still gives us a good idea of a few metrics.

1. Our model did not suffer from mode collapse. Despite being unable to generate the deeper features such as individual textures, eyes, nose, ears etc we do not see any repeated patterns.  

 ```
  
  
 ```
 ## Install Requirements: 
 
 The following were used for making this program-
 1. Tensorflow gpu
 2. sklearn
 3. numpy
 4. pandas
 5. os module
 6. unittest
 
 ```
 pip install -r requirements.txt
 
 ```
The following link provides a good walkthrough to setup tensorflow:

```
https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
```

## Format code to PEP-8 standards (Important for contributing to the repo): 
 
 This repository is strictly based on *PEP-8* standards. To assert PEP-8 standards after editing your own code, use the following: 
 
 ```
 black DCGAN.py

 ```
 
If you wish to change the dataset used here change the following to correctly reflect the directory in `DCGAN.py` :

`DIR = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\GANs\PokemonDataset\pokemon_jpg\pokemon_jpg"`


NOTE: Due to slow training on collab this model was trained on a physical 2080Super gpu which has a limited vram of 8gb. This was done using tensorflow GPU, images were resized to fit vram constraints. Training will take longer on GPUs not running CUDA or if larger datasets are used or if images are not resized. Do NOT attempt to train this on a CPU as it may freeze or crash your system/runtime-session.

 
### Reference: 

1. https://arxiv.org/abs/1406.2661 (Generative Adversarial Networks, Goodfellow et al.)
2. https://www.casualganpapers.com/
3. https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
4. https://arxiv.org/pdf/1511.06434.pdf (Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Alec et al.)
 
