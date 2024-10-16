# Edge Detection via Convolution: Three Approaches
This project demonstrates convolution-based edge detection on a sample image, implemented using three different methods to explore convolution functionality, implementation challenges, and its role in image analysis.

**Methods:**

Method 1: Utilizing scipy.signal.convolve2d.
Method 2: Manual implementation with nested for-loops for a deeper understanding.
Method 3: Using PyTorch's conv2d layer with a hardcoded kernel.

**Constraints:**

1) Stride of 1.
2) Input image dimensions remain unchanged after convolution, achieved via zero-padding.
3) Input images are cast to double precision before filtering.
4) The provided kernel (KERNEL) is used for all methods.
   
The execution time of each method is measured and compared to highlight performance differences.
