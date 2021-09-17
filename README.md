# Computer-Vision

This repository contains all projects and coursework done in my Computer Vision course.
In some of these projects, I have created custom functions that imitate the functionality of OpenCV functions. Other projects include implementing the functionality of the 'CamScanner' app, panorama stitching, etc.
Some of the end results of these projects have been demonstrated below:

## 1. Image Filtering and Hybrid Images
 Here, I have implemented the custom <i>my_imfilter()</i> function which imitates the <i>filter2D</i> function of the OpenCV library. A hybrid image is the sum of a low-pass filtered version of one image and a high-pass filtered version of a second image. A low-pass filtered image is obtained by applying convolution operation on an image by a Gaussian kernel (ie, a low pass filter), and a high-pass filtered image is obtained by subtracting the low-pass filtered image from the original image.
<p float="left">
  <img src="/Images/low_frequencies.jpg" width="250" /> &emsp;
  <img src="/Images/high_frequencies.jpg" width="250" /> 
</p>
<p>
    <p>&ensp;Low-pass filtered image of a dog &emsp;&emsp;&ensp;
 High-pass filtered image of a cat</p>
</p>
<img src="/Images/pyramid.jpg" width="550" /> 
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Hybrid image</p>
One more example:
<img src="/Images/pyramid2.png" width="550" /> 
