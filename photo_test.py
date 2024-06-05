import numpy as np
from matplotlib import pyplot as plt

img = plt.imread('./data/img_1.png')
# print(img.shape)
img = img[..., :3]

img = np.uint8(np.dot(img,[0.3,0.3,0.4]))
print(img.shape)

l1 = 28/img.shape[0]
l2 = 28/img.shape[1]
new_img = np.zeros((28,28))
for x in range(28):
    for y in range(28):
        new_img[x][y] = img[int(x/l1)][int(y/l2)]

# plt.imshow(new_img, cmap ='gray')
# plt.show()
z = np.array(new_img)
z = z.reshape((1,28,28))
plt.imshow(new_img, cmap ='gray')
plt.show()


# import numpy as np
# from matplotlib import pyplot as plt
#
# img = plt.imread('./data/img_1.png')
#
# # Convert RGBA to RGB
# rgb_img = img[..., :3]
#
# plt.imshow(rgb_img)
# plt.show()