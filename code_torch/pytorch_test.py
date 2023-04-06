import torch
img = torch.zeros((5, 1, 6, 6))
# print(img)

img2 = torch.zeros((5, 3, 6, 6))
print(img2)

im3 = torch.cat([img, img, img], dim = 1)
print(im3.size())
print(im3)
# img = img.expand(3,*img.shape[1:]) 

# print(img)
# print(img.size())