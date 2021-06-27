import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)
  
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  p1 = filter.shape[0]//2
  p2 = filter.shape[1]//2
  pad1 = np.pad(image[:,:,0],((p1,p1),(p2,p2)),'reflect')
  pad2 = np.pad(image[:,:,1],((p1,p1),(p2,p2)),'reflect')
  pad3 = np.pad(image[:,:,2],((p1,p1),(p2,p2)),'reflect')
  image_pad = np.dstack([pad1,pad2,pad3])
  filtered_image = np.zeros((image.shape[0],image.shape[1],3))
  for c in range(3):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i,j,c] = np.sum(image_pad[i:i+filter.shape[0],j:j+filter.shape[1],c]*filter)
                                    
  return filtered_image


def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1,filter)
  im2blur = my_imfilter(image2,filter)
  high_frequencies = image2 - im2blur
  hybrid_image = low_frequencies + high_frequencies 

  return low_frequencies, high_frequencies, hybrid_image
