# Color Classification using a Single Gaussian
In this project, we explore object segmentation in images using a probabilistic approach based on color. Rather than deterministically labeling a pixel as a specific color (e.g., red or green), we assign probabilities to different colors. This method is particularly useful in real-world scenarios where sensor noise and varying lighting conditions can affect color perception.

We treat color classification as a machine learning problem, where each pixel is given a probability of belonging to a particular color class (e.g., determining the likelihood that a pixel belongs to a green cap based on its RGB values). Given a pixel’s RGB values, we estimate the probability that it belongs to a specific class.

For each class, we estimate the spread of RGB values by analyzing the dataset to model the probability distribution. To compute the likelihood of a pixel belonging to a particular class given its RGB value, we apply Bayes' rule.

# Approach 
1) Estimate the mean and covariance matrix for each class after extract the RGB values from each image corresponding to that particular class.
2) Compute the probability density function for every pixel in the provided images. Then make a binary mask using a threshold (tunable).
3) Apply the binary mask on the provided images to segment objects of each class separately.

We perform color segmentation to identify and classify four distinct objects from RGB images:

**Class 0:** Green cap (smallest)
**Class 1:** Yellow cap
**Class 2:** Blue cap
**Class 3:** Red cap (largest)

We work with 7 RGB images that contain these objects, and the task involves using Gaussian models to represent the probability distribution of each class. By modeling the color distributions, we can accurately segment the objects based on their RGB values.

# Implementation (Pseudo Code for Python)
**Training Code**
Rerun this code for each class separately:

Initialize an empty list/matrix to store RGB values from all images.

For each image index from 1 to 7:

Read the image from the file path.
1) Display the image (using imshow).
2) Draw a freehand shape on the image and create a mask for this shape (use roipoly in MATLAB or find/use any equivalent function in Python).
3) Extract the RGB channels from the image.
4) Apply the mask to get RGB values from the selected area.
5) Combine the RGB values into a single list.
6) Append this list to the main list of RGB values.
  
Estimate the mean of the RGB values.
Estimate the covariance of the RGB values.

**Color Segmentation for a Single Class**
For each image index from 1 to 10:

Load the image from the file.
1) Apply a Gaussian distribution to the image using the applyGaussianToImage function.
2) Create a mask where the probability is greater than 1e-6.
3) Apply the mask.
   
Function applyGaussianToImage:
1) Convert the image to double precision.
2) Reshape the image into a matrix where each row is a pixel vector (R, G, B).
3) Compute the probability density function (PDF) for each pixel using a multivariate Gaussian distribution.
4) Reshape the PDF result back to image dimensions.
5) Return the resulting image with PDF values.

 
