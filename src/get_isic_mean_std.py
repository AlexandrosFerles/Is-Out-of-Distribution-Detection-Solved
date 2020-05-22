from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

path = '/raid/ferles/ISIC2019/ISIC_2019_Training_Input'
dataset = ImageFolder(path, transform=transform)
loader = DataLoader(dataset, batch_size=20)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)