import os, numpy, PIL
from PIL import Image

dir = 'Images/normal_dice/10'
# Access all PNG files in directory
allfiles = os.listdir(dir)
imlist = [filename for filename in allfiles if filename[-4:] in [".jpg",".PNG"]]

print(imlist)
# Assuming all images are the same size, get dimensions of first image
print(imlist[0])
w,h=Image.open(dir+'/'+imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w),numpy.float64)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr = numpy.array(Image.open(dir +'/'+ im),dtype=numpy.float64)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr)
out.save("Average_faceLogo_Rotate3.png")
out.show()