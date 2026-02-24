
import torch
import torch.nn
import torch.nn.functional
import torchvision
import PIL.Image
import pathlib
import matplotlib.pyplot



print("all libs are loaded" + "-"*50)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
])
device = torch.device("cpu")

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

loader = torch.utils.data.DataLoader(my_dataset, shuffle = True, batch_size = 39)

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)  
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(32)  
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(32)
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(32)
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(32)
        
        self.transpose_conv1 = torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.transpose_conv2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.transpose_conv3 = torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(32)
        self.transpose_conv4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = torch.nn.functional.relu(self.bn6(self.conv6(x)))
        x = torch.nn.functional.relu(self.bn7(self.conv7(x)))
        x = torch.nn.functional.relu(self.bn8(self.conv8(x)))
        
        x = torch.nn.functional.relu(self.bn4(self.transpose_conv1(x)))
        x = torch.nn.functional.relu(self.bn5(self.transpose_conv2(x)))
        x = torch.nn.functional.relu(self.bn6(self.transpose_conv3(x)))
        x = torch.nn.functional.relu(self.transpose_conv4(x))

        return x
   
model = net()
model = model.to(device)
print("model loaded" + "-"*50)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
'''
matplotlib.pyplot.ion()  # 开启交互模式
fig, ax = matplotlib.pyplot.subplots()
x_data, y_data = [], []
iid = 0

for i in range(0, 20):
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
        y_data.append(total_loss/10/(id + 1))
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
model.eval()
image1 = my_dataset[0][0].permute(1,2,0)
image11 = my_dataset[0][1].permute(1,2,0)
image2 = model(my_dataset[0][0].unsqueeze(0))[0].permute(1,2,0).detach()
'''for i in image2:
    for j in i:
        if j != 0:
            j *= 50
            '''
fig, axes = matplotlib.pyplot.subplots(1,3)
axes[0].imshow(image1)
axes[1].imshow(image11)
axes[2].imshow(image2)
matplotlib.pyplot.show()

