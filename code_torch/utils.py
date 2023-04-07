import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    npimg = img.cpu()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()