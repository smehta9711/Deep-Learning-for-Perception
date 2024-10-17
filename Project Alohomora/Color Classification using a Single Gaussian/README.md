# Color Classification using a Single Gaussian
In this project, we explore object segmentation in images using a probabilistic approach based on color. Rather than deterministically labeling a pixel as a specific color (e.g., red or green), we assign probabilities to different colors. This method is particularly useful in real-world scenarios where sensor noise and varying lighting conditions can affect color perception.

We treat color classification as a machine learning problem, where each pixel is given a probability of belonging to a particular color class (e.g., determining the likelihood that a pixel belongs to a green cap based on its RGB values). Given a pixel’s RGB values, we estimate the probability that it belongs to a specific class.

For each class, we estimate the spread of RGB values by analyzing the dataset to model the probability distribution. To compute the likelihood of a pixel belonging to a particular class given its RGB value, we apply Bayes' rule.

# Approach 
1) Estimate the mean and covariance matrix for each class after extract the RGB values from each image corresponding to that particular class.
2) Compute the probability density function for every pixel in the provided images. Then make a binary mask using a threshold (tunable).
3) Apply the binary mask on the provided images to segment objects of each class separately.

 {
   "cell_type": "markdown",
   "id": "5c586079",
   "metadata": {},
   "source": [
    "## 4. Implementation (Psuedo Code for Python) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e9901fd",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# Training Code\n",
    "\n",
    "Rerun this code for each class separately\n",
    "\n",
    "Initialize an empty list/matrix to store RGB values from all images\n",
    "\n",
    "For each image index from 1 to 7:\n",
    "    - Read the image from the file path\n",
    "    - Display the image (imshow)\n",
    "\n",
    "    - Draw a freehand shape on the image and create a mask for this shape (`roipoly` in MATLAB, find or use any function for this in Python)\n",
    "\n",
    "    - Extract the RGB channels from the image\n",
    "    - Apply the mask to get RGB values from the selected area\n",
    "\n",
    "    - Combine the RGB values into a single list\n",
    "    - Append this list to the main list of RGB values\n",
    "\n",
    "Estimate the mean of the rgb values\n",
    "Estimate the covariance of the RGB values\n",
    "\n",
    "# Color Segmentation for a single class\n",
    "\n",
    "For i from 1 to 10:\n",
    "    - Load the image from the file\n",
    "    - Apply Gaussian distribution to the image using `applyGaussianToImage` function\n",
    "    - Create a mask where the probability is greater than 1e-6\n",
    "    - Apply the mask\n",
    "\n",
    "Function `applyGaussianToImage`:\n",
    "    - Convert image to double precision\n",
    "    - Reshape the image into a matrix where each row is a pixel vector (R, G, B)\n",
    "    - Implement and compute the probability density function (PDF) for each pixel using multivariate Gaussian \n",
    "    - Reshape the PDF result back to image dimensions\n",
    "    - Return the resulting image with PDF values"
   ]
  },
