# %%
import torch.nn.functional as F

def up(inp,target):
    return F.interpolate(inp,size=target.shape[2:],mode="bilinear",align_corners=True),inp

# %%
import torch
class Encoder(torch.nn.Module):
    def __init__(self,i_channel,o_channel,kernel=3):
        super().__init__()
        self.Seq=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=i_channel,
                            kernel_size=kernel,
                            out_channels=o_channel,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=o_channel,
                            kernel_size=kernel,
                            out_channels=o_channel,
                            padding=1),
            torch.nn.ReLU()
        )
        self.Pool=torch.nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        #essa esliya liya kyunki Seq ke output ko middle layers lo bhi pass
        #karna hai aur lower level ko bhi aur lower level me bass pool chahiye side me nahi
    def forward(self,inp):
        skip=self.Seq(inp)
        out=self.Pool(skip)
        return out,skip



# %%
class Decoder(torch.nn.Module):
    def __init__(self,i_channel,skip_channel,o_channel,kernel=3):
        super().__init__()
        self.Seq1=torch.nn.Sequential(
               torch.nn.Upsample(scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True),
               torch.nn.Conv2d(in_channels=i_channel,
                               out_channels=o_channel,
                               kernel_size=kernel,
                               padding=1),
               torch.nn.ReLU()
        )
        self.Seq2=torch.nn.Sequential(
               torch.nn.Conv2d(in_channels=o_channel+skip_channel,
                               out_channels=o_channel,
                               kernel_size=kernel,
                               padding=1),
               torch.nn.ReLU(),
               torch.nn.Conv2d(in_channels=o_channel,
                               out_channels=o_channel,
                               kernel_size=kernel,
                               padding=1),
               torch.nn.ReLU()
        )
    def forward(self,inp,skips):
        upgrade=self.Seq1(inp)
        con = torch.concat([upgrade]+skips, dim=1)
        out = self.Seq2(con)
        return out

# %%
class Midlayer(torch.nn.Module):
    def __init__(self, i_channel, o_channel,kernel=3):
        super().__init__()
        self.Seq=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=i_channel,out_channels=o_channel,kernel_size=kernel,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=o_channel,out_channels=o_channel,kernel_size=kernel,padding=1),
            torch.nn.ReLU()
        )
    def forward(self,inputs):
        con_cat=torch.cat(inputs,dim=1)
        return self.Seq(con_cat)

# %%
'''
layer1_out_channel=64,
layer3_out_channel=128,
layer3_out_channel=256,
layer4_out_channel=512,
final_layer_out_channel=1024,
'''
class UnetPP(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.X00=Encoder(3,64)
        self.X10=Encoder(64,128)
        self.X01=Midlayer(64+128,64)      #i_channel=X00+X10 o_channel=layer_out_channels)
        self.X20=Encoder(128,256)
        self.X11=Midlayer(128+256,128)
        self.X02=Midlayer(64+64+128,64)    #i_channel=X00+X11,X01 o_channel=layer2_out_channels)
        self.X30=Encoder(256,512)
        self.X21=Midlayer(256+512,256)
        self.X12=Midlayer(128+256+128,128)
        self.X03=Midlayer(64+64+64+128,64)
        self.X40=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,
                            kernel_size=3,
                            out_channels=1024,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=1024,
                            kernel_size=3,
                            out_channels=1024,
                            padding=1),
            torch.nn.ReLU()
        )
        self.X31=Decoder(1024,512,512)
        self.X22=Decoder(512,256*2,256)
        self.X13=Decoder(256,128*3,128)
        self.X04=Decoder(128,64*4,64)
        self.final=torch.nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size=1,
                                    padding=0)
    def forward(self,inp):
        out,skip0_0=self.X00(inp)
        out,skip1_0=self.X10(out)
        skip0_1=self.X01([up(skip1_0,skip0_0)])
        out,skip2_0=self.X20(out)
        skip1_1=self.X11([up(skip2_0,skip1_0)])
        skip0_2=self.X02([skip0_0,up(skip1_1,skip0_1)])
        out,skip3_0=self.X30(out)
        skip2_1=self.X21([up(skip3_0,skip2_0)])
        skip1_2=self.X12([skip1_0,up(skip2_1,skip1_1)])
        skip0_3=self.X03([skip0_0,skip0_1,up(skip1_2,skip0_2)])
        out=self.X40(out)
        out=self.X31(out,[skip3_0])
        out=self.X22(out,[skip2_1,skip2_0])
        out=self.X13(out,[skip1_2,skip1_1,skip1_0])
        out=self.X04(out,[skip0_3,skip0_2,skip0_1,skip0_0])
        out=self.final(out)
        return out

# %%
x = torch.randn(1, 3, 256, 256)
model = UnetPP()
y = model(x)
print(y.shape)


