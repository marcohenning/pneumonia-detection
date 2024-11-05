from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import PneumoniaDataset


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((1,), (1,))
])
dataset = PneumoniaDataset(transform)

batch_size = 32

training_split = int(0.75 * len(dataset))
testing_split = len(dataset) - training_split
training_data, testing_data = random_split(dataset, [training_split, testing_split])
training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
