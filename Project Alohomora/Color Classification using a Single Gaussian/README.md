# Color Classification using a Single Gaussian
In this project, we explore object segmentation in images using a probabilistic approach based on color. Rather than deterministically labeling a pixel as a specific color (e.g., red or green), we assign probabilities to different colors. This method is particularly useful in real-world scenarios where sensor noise and varying lighting conditions can affect color perception.

We treat color classification as a machine learning problem, where each pixel is given a probability of belonging to a particular color class (e.g., determining the likelihood that a pixel belongs to a green cap based on its RGB values). Given a pixel’s RGB values, we estimate the probability that it belongs to a specific class.

For each class, we estimate the spread of RGB values by analyzing the dataset to model the probability distribution. To compute the likelihood of a pixel belonging to a particular class given its RGB value, we apply Bayes' rule
