
import imageio
import matplotlib.pyplot as plt
import numpy as np  
import skimage
import pdb
from skimage.transform import rescale

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect',multichannel= True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

Im_papa = imageio.imread("./imgs/papa.jpg")
Im_yo = imageio.imread("./imgs/yo.jpg")

G_papa = skimage.filters.gaussian(Im_papa,sigma = 3,multichannel = True, preserve_range=True)
G_yo = skimage.filters.gaussian(Im_yo,sigma = 19,multichannel = True, preserve_range=True)

HP = np.float64(Im_papa) - G_papa
LP = G_yo.copy()
Hyb = np.uint8(LP+HP)

fig = plt.figure()
plt.subplot(121)
plt.imshow(Hyb)

Vis = vis_hybrid_image(Hyb)
plt.subplot(122)
plt.imshow(Vis)

plt.show()

skimage.io.imsave("./imgs/hybrid.png",Hyb)