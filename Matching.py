from PIL import Image
import imagehash
hash0 = imagehash.average_hash(Image.open('Average_face1.png'))
#Try to increase the contrast of this image make this with OpenCV.
hash1 = imagehash.average_hash(Image.open('Images/anomalous_dice/img_17885_cropped.jpg',0))
cutoff = 2  # maximum bits that could be different between the hashes.

if hash0 - hash1 < cutoff:
  print('images are similar')
else:
  print('images are not similar')