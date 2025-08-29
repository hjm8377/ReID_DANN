import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils as utils

import torch.utils.data as data

import torchvision.models as models

import torchvision.utils as v_utils

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

# %matplotlib inline

 

content_layer_num = 1

image_size = 512

epoch = 5000

 

content_dir = "D:\\ReID\\NightReID\\bounding_box_train\\0001R1C001.jpg"

style_dir = "D:\\ReID\\market1501\\bounding_box_train\\0002_c1s1_000451_03.jpg"




def image_preprocess(img_dir):

    img = Image.open(img_dir)

    transform = transforms.Compose([

                    transforms.Resize(image_size),

                    transforms.CenterCrop(image_size),

                    transforms.ToTensor(),

                    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], 

                                        std=[1,1,1]),

                    #transforms.Normalize([0.5], [0.5])

                ])

    img = transform(img).view((-1,3,image_size,image_size))

    return img




def image_postprocess(tensor):

    transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], 

                                     std=[1,1,1])

    #transform = transforms.Normalize([0.5], [0.5])

    img = transform(tensor.clone())

    img = img.clamp(0,1)

    img = torch.transpose(img,0,1)

    img = torch.transpose(img,1,2)

    return img

 

resnet = models.resnet50(pretrained=True)

for name,module in resnet.named_children():

    print(name)




class Resnet(nn.Module):

    def __init__(self):

        super(Resnet,self).__init__()

        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])

        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])

        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])

        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])

        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])

        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

 

    def forward(self,x):

        out_0 = self.layer0(x)

        out_1 = self.layer1(out_0)

        out_2 = self.layer2(out_1)

        out_3 = self.layer3(out_2)

        out_4 = self.layer4(out_3)

        out_5 = self.layer5(out_4)

        return out_0, out_1, out_2, out_3, out_4, out_5




class GramMatrix(nn.Module):

    def forward(self, input):

        b,c,h,w = input.size()

        F = input.view(b, c, h*w)

        G = torch.bmm(F, F.transpose(1,2)) 

        return G

 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

 

resnet = Resnet().to(device)

for param in resnet.parameters():

    param.requires_grad = False    





class GramMSELoss(nn.Module):

    def forward(self, input, target):

        out = nn.MSELoss()(GramMatrix()(input), target)

        return out




content = image_preprocess(content_dir).to(device)

style = image_preprocess(style_dir).to(device)

generated = content.clone().requires_grad_().to(device)

 

print(content.requires_grad,style.requires_grad,generated.requires_grad)

 

plt.imshow(image_postprocess(content[0].cpu()))

plt.show()

 

plt.imshow(image_postprocess(style[0].cpu()))

plt.show()

 

gen_img = image_postprocess(generated[0].cpu()).data.numpy()

plt.imshow(gen_img)

plt.show()




style_target = list(GramMatrix().to(device)(i) for i in resnet(style))

content_target = resnet(content)[content_layer_num]

style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

 

optimizer = optim.LBFGS([generated])

 

iteration = [0]

while iteration[0] < epoch:

    def closure():

        optimizer.zero_grad()

        out = resnet(generated)

 

        style_loss = [GramMSELoss().to(device)(out[i],style_target[i])*style_weight[i] for i in range(len(style_target))]

 

        content_loss = nn.MSELoss().to(device)(out[content_layer_num],content_target)

 

        total_loss = 1000 * sum(style_loss) + torch.sum(content_loss)

        total_loss.backward()

 

        if iteration[0] % 100 == 0:

            print(total_loss)

        iteration[0] += 1

        return total_loss

 

    optimizer.step(closure)

 

gen_img = image_postprocess(generated[0].cpu()).data.numpy()

plt.figure(figsize=(10,10))

plt.imshow(gen_img)

plt.show()

plt.savefig('nvh_gen.png')

#gen_img.save("drive/MyDrive/ST.jpg")