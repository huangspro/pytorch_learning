
import torch
import torch.nn
import torch.nn.functional
import torchvision
import PIL.Image
import pathlib
import matplotlib.pyplot

batch = 8
device = torch.device("cuda")

torch.cuda.empty_cache()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        X = pathlib.Path("../data/")
        Y = pathlib.Path("../data_test/")
        self.X_data = [str(i) for i in X.iterdir()]
        self.Y_data = [str(i) for i in Y.iterdir()]
        
    def __len__(self):
        return len(self.X_data)
        
    def __getitem__(self, idx):
        return (transform(PIL.Image.open(self.X_data[idx]).convert('L')), transform(PIL.Image.open(self.Y_data[idx]).convert('L')))
        
my_dataset = MyDataset()
print("data loaded" + "-"*50)

loader = torch.utils.data.DataLoader(my_dataset, shuffle = True, batch_size = batch)

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64)

        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)

        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(128)

        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(128)

        self.conv8 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(256)

        self.conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = torch.nn.BatchNorm2d(256)

        self.conv10 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = torch.nn.BatchNorm2d(256)

        self.conv11 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = torch.nn.BatchNorm2d(256)

        self.conv12 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = torch.nn.BatchNorm2d(512)

        self.conv13 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = torch.nn.BatchNorm2d(512)

        self.conv14 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn14 = torch.nn.BatchNorm2d(512)

        self.conv15 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn15 = torch.nn.BatchNorm2d(512)

        self.conv16 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn16 = torch.nn.BatchNorm2d(512)

        # -------- Decoder --------
        self.tr1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn17 = torch.nn.BatchNorm2d(256)

        self.tr2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn18 = torch.nn.BatchNorm2d(128)

        self.tr3 = torch.nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        self.bn19 = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        # -------- Encoder --------
        x =  torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x =  torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x =  torch.nn.functional.max_pool2d( torch.nn.functional.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2) 
        x =  torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x =  torch.nn.functional.max_pool2d( torch.nn.functional.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2)  
        x =  torch.nn.functional.relu(self.bn6(self.conv6(x)))
        x =  torch.nn.functional.relu(self.bn7(self.conv7(x) + x))
        x =  torch.nn.functional.max_pool2d( torch.nn.functional.relu(self.bn8(self.conv8(x))), kernel_size=2, stride=2)  
        x =  torch.nn.functional.relu(self.bn9(self.conv9(x)))
        x =  torch.nn.functional.relu(self.bn10(self.conv10(x) + x))
        x =  torch.nn.functional.relu(self.bn11(self.conv11(x) + x))
        x =  torch.nn.functional.relu(self.bn12(self.conv12(x)))
        x =  torch.nn.functional.relu(self.bn13(self.conv13(x) + x))
        x =  torch.nn.functional.relu(self.bn14(self.conv14(x) + x))
        x =  torch.nn.functional.relu(self.bn15(self.conv15(x) + x))
        x =  torch.nn.functional.relu(self.bn16(self.conv16(x) + x))

        # -------- Decoder --------
        x =  torch.nn.functional.relu(self.bn17(self.tr1(x)))  
        x =  torch.nn.functional.relu(self.bn18(self.tr2(x)))  
        x =  torch.nn.functional.relu(self.bn19(self.tr3(x))) 

        return x
   
# model = net()
model = torch.load("model.pth", weights_only = False)
model = model.to(device)
print("model loaded" + "-"*50)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.000002)
'''
matplotlib.pyplot.ion()  # 开启交互模式
fig, ax = matplotlib.pyplot.subplots()
x_data, y_data = [], []
iid = 0

for i in range(0, 10):
    total_loss = 0
    for id,(x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        x_data.append(iid)
        y_data.append(total_loss/batch/(id + 1))
        ax.clear()
        ax.plot(x_data, y_data, 'b-o')
        ax.set_xlim(0, max(10, iid))
        ax.set_ylim(0, max(y_data))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        matplotlib.pyplot.pause(1)
        iid += 1
    print(f"epoch: {i}, loss: {total_loss/len(my_dataset)}")
    if total_loss/len(my_dataset) < 1e-6:
        break
torch.save(model, "model.pth")
matplotlib.pyplot.ioff()
matplotlib.pyplot.show()

'''
#test the input and output
model = torch.load("model.pth", weights_only = False)
model = model.to(device)
model.eval()

index = 5

image1 = my_dataset[index][0].permute(1,2,0)
image11 = my_dataset[index][1].permute(1,2,0)
#image2 = model(my_dataset[index][0].unsqueeze(0).to(device))[0].permute(1,2,0).detach().cpu()
image2 = model(transform(PIL.Image.open("ok.jpg").convert('L')).unsqueeze(0).to(device))[0].permute(1,2,0).detach().cpu()

        
fig, axes = matplotlib.pyplot.subplots(1,3)
axes[0].imshow(image1)
axes[1].imshow(image11)
axes[2].imshow(image2, cmap = 'hot')
matplotlib.pyplot.show()

