import torch
import torch.nn
import torch.nn.functional
import torchvision
import PIL.Image
import pathlib
import matplotlib.pyplot
class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)  
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
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
        
grad = 0

def hook(module, input, output):
    global grad
    grad = output
    
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
    #torchvision.transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
])

criterion = torch.nn.CrossEntropyLoss()
model = torch.load("model.pth")
model.conv3.register_backward_hook(hook)

x = transform(PIL.Image.open('../pet_data/basset_hound_3.jpg').convert('RGB'))

fig, axe = matplotlib.pyplot.subplots(1,2)
axe[0].imshow(x.permute(1,2,0))
x = x.unsqueeze(0)

output = model(x)
loss = criterion(output, torch.tensor([0]))
loss.backward()

y = grad[0][0]
out = torch.zeros(64,64)
for i in y:
    out = out + i
axe[1].imshow(out, cmap = 'hot')

matplotlib.pyplot.show()
