# Facial Emotion Recognition

## Abstract
The goal of this project was to use deep learning neural networks to recognize facial emotions from images such as happiness, sadness, anger... etc.

## Data
The data was downloaded from **Kaggle** which contains images of faces displaying a variety of emotions.
- 37,303 total images split into 7 classes:
    1. 5,500 images (Happiness)
    2. 5,500 images (Anger)
    3. 4,303 images (Disgust)
    4. 5,500 images (Fear)
    5. 5,500 images (Neutral)
    6. 5,500 images (Sadness)
    7. 5,500 images (Surprise)

## Design

*Image Augmentation*

Resizing and rescaling the images.
Flipping images horizontally.
Rotating images up to 20°.

*Transfer Learning*
Using pretrained neural networks and adding more layers to them to make use of the trained weights of said neural networks. Networks used are DenseNet, VGG16, VGG19, Inception, and ResNet.


## Tools

Software Platform
- Jupyter Notebook

Programming Language
- Python
 
Python Libraries:
- Machine learning libraries:
  - Tensorflow
  - Keras


- Data manipulation libraries:
  - Pandas
  - Numpy


- Visualization libraries:
  - Matplotlib
  - Seaborn
 

## Communication
Slides containing visualizations
