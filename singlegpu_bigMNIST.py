import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_train_objs(data_dir='/scratch/g/bgross/bigMNIST'):
    """Loads MNIST with augmentation to increase dataset size, saving to disk."""
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Define augmentation techniques
    aug_transforms = [
        transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        transforms.Compose([
            transforms.ElasticTransform(alpha=30.0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ]

    # Check if MNIST exists, download if not.
    download_flag = not os.path.exists(os.path.join(data_dir, 'MNIST', 'raw'))
    
    # Load original MNIST dataset
    base_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=download_flag, transform=base_transform)

    # Create augmented versions
    augmented_datasets = [torchvision.datasets.MNIST(data_dir, train=True, download=False, transform=aug) for aug in aug_transforms] #Download is false as it is already downloaded

    # Combine datasets
    num_copies = 200  # Number of times to repeat the pattern
    expanded_dataset = ConcatDataset(( [base_dataset] + augmented_datasets ) * num_copies)
   
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return expanded_dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)

class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            del source, targets # remove from GPU memory
            torch.cuda.empty_cache() # clear cache
    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    device = 0
    main(device, args.total_epochs, args.save_every, args.batch_size)