import torch.nn as nn

class Generator(nn.Module):

    def __init__(self,latent_dim,img_size,channels):
        super(Generator,self).__init__()

        self.init_size = img_size//4
        self.latent_dim = latent_dim
        self.outputchannels = channels
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim,128*self.init_size**2))


        self.conv_net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,self.outputchannels,3,stride=1,padding=1),
            nn.Tanh(),
        )

    def forward(self,z):
        out = self.l1(z)
        out = out.view(out.shape[0],128,self.init_size,self.init_size)
        img = self.conv_net(out)

        return img

class Discriminator(nn.Module):

    def __init__(self,img_size,channels):
        super(Discriminator,self).__init__()

        self.channels = channels
        self.img_size = img_size

        def disciminator_block(in_filters, out_filters,bn=True):

            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]

            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *disciminator_block(self.channels,16, bn=False),
            *disciminator_block(16,32),
            *disciminator_block(32,64),
            *disciminator_block(64,128),
        )

        ds_size = self.img_size//2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2,1),nn.Sigmoid())

    def forward(self,img):

        out = self.model(img)
        out = out.view(out.shape[0],-1)
        validity = self.adv_layer(out)

        return validity
