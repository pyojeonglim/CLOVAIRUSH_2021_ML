from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    print(np_img.shape)
    print((np.transpose(np_img,(1,2,0))).shape)

trans = transforms.Compose([transforms.Resize((100,100)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           ])
trainset = torchvision.datasets.ImageFolder(root = "./clothes",
                                           transform = trans)

trainset.__getitem__(18)
len(trainset)
trainloader = DataLoader(trainset,
                        batch_size = 16,
                        shuffle = False,
                        num_workers = 4)
classes = trainset.classes
classes
['bag', 'shoes', 'tshirts']
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)
tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
print(images.shape)
imshow(torchvision.utils.make_grid(images, nrow=4))
print(images.shape)
print((torchvision.utils.make_grid(images)).shape)
print("".join("%5s " % classes[labels[j]] for j in range(16)))
torch.Size([16, 3, 100, 100])
torch.Size([16, 3, 100, 100])
torch.Size([3, 206, 818])
