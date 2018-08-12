# Automatic Image Colorization
### Introduction
This project implements a deep convolutional neural network for automatic colorization, the problem of converting grayscale input images into colored images. The model is based on the [ResNet-18](https://arxiv.org/abs/1512.03385) classifier and trained on the [MIT Places365](http://places2.csail.mit.edu/) database of landscapes and scenes. It is inspired by previous work from [Iizuka et al.](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/), [Zhang et al.](http://richzhang.github.io/colorization/), and [Larsson et al.](http://people.cs.uchicago.edu/~larsson/colorization/) on image colorization. Instructions for colorizing images using the provided pre-trained weights or training the network from scratch may be found below. All code is open-source and implementation details are described in the project report.

Feel free to reach out to me (lukemelas) for any questions regarding the model!

# Demos

Here are the results of the network on validation images from the Places365 Dataset:
![Colorization Results](https://github.com/lukemelas/Automatic-Image-Colorization/blob/master/results.jpg)

Here, we use the network to colorize a black-and-white Charlie Chaplin film. The model is applied frame-by-frame, with no temporal smoothing applied:

[![Colorizing Charlie Chaplin](https://github.com/lukemelas/Automatic-Image-Colorization/blob/master/charlie.jpg)](https://www.youtube.com/watch?v=LluZarKPY-o)
