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
    #torchvision.transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
])
device = torch.device("cpu")

print("transformer and cuda loaded" + "-"*50)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        path = "../pet_data/"
        All_Data = pathlib.Path(path)
        self.X_data = [str(i) for i in All_Data.iterdir()]
        self.Y_data = []
        tem = ''
        index = -1
        for i in All_Data.iterdir():
            if tem != str(i).split('_')[0]:
                tem = str(i).split('_')[0]
                index += 1
            self.Y_data.append(index)
        
    def __len__(self):
        return len(self.X_data)
        
    def __getitem__(self, idx):
        return (transform(PIL.Image.open(self.X_data[idx]).convert('RGB')), torch.tensor(self.Y_data[idx]))
        
my_dataset = MyDataset()
print("data loaded" + "-"*50)

loader = torch.utils.data.DataLoader(my_dataset, shuffle = True, batch_size = 64)

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=10, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)  
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=10, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=10, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.linear1 = torch.nn.Linear(128*32*32, 128)
        self.linear2 = torch.nn.Linear(128, 100)
        self.linear3 = torch.nn.Linear(100, 50)
        self.linear4 = torch.nn.Linear(50, 50)

        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)
        
        x = x.view(-1, 128*32*32)
        
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
model = net()
model = model.to(device)
print("model loaded" + "-"*50)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for i in range(0, 8):
    total_loss = 0
    for _,(x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch: {i}, loss: {total_loss/len(my_dataset)}")
    
    
torch.save(model, "model.pth")
