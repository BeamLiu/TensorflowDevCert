import scipy.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

acent_image = datasets.ascent()
plt.imshow(acent_image)
plt.gray()
plt.show()

image_transformed = np.copy(acent_image)
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]
print(f'image size: {size_x}, {size_y}')

#filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun
#filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# If all the digits in the filter don't add up to o or 1, you should probably do a weight to get it to do so# so,for example,if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight = 1

# Iterate over the image
# filter
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0.
        convolution = convolution + (acent_image[x - 1, y - 1] * filter[0][0])
        convolution = convolution + (acent_image[x - 1, y] * filter[0][1])
        convolution = convolution + (acent_image[x - 1, y + 1] * filter[0][2])
        convolution = convolution + (acent_image[x, y - 1] * filter[1][0])
        convolution = convolution + (acent_image[x, y] * filter[1][1])
        convolution = convolution + (acent_image[x, y + 1] * filter[1][2])
        convolution = convolution + (acent_image[x + 1, y - 1] * filter[2][0])
        convolution = convolution + (acent_image[x + 1, y] * filter[2][1])
        convolution = convolution + (acent_image[x + 1, y + 1] * filter[2][2])

        convolution *= weight

        # Relu
        convolution = convolution if convolution > 0 else 0

        image_transformed[x][y] = convolution

plt.imshow(image_transformed)
plt.show()

# pool
new_x = int(size_x / 2)
new_y = int(size_y / 2)
new_image = np.zeros((new_x, new_y))
f, axs = plt.subplots(1, 2)
# Iterator over the image
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = [image_transformed[x][y], image_transformed[x + 1][y], image_transformed[x][y + 1],
                  image_transformed[x + 1][y + 1]]
        new_image[int(x / 2), int(y / 2)] = max(pixels)
axs[0].imshow(image_transformed)
axs[0].set_title("Filtered Image")
axs[1].imshow(new_image)
axs[1].set_title("Pooled Image")

plt.show()
