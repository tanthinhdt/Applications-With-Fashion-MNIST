import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tforms
import torchvision.datasets as dtsets
from tqdm import tqdm


class Training():
    def __init__(self, data_dir, ckpt_dir, train_ratio, interpolate, 
                 img_size, model, optimizer, scheduler=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.train_ratio = train_ratio
        self.interpolate = interpolate
        self.img_size = img_size

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_history = {
            'train_loss': [],
            'val_acc': []
        }

    def load_data(self):
        if self.interpolate:
            transforms = tforms.Compose([
                tforms.ToTensor(),
                tforms.Resize((32, 32)),
                tforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        else:
            transforms = tforms.Compose([
                tforms.ToTensor(),
                tforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])

        train_set = dtsets.FashionMNIST(
            root=self.data_dir, download=True, train=True, transform=transforms)

        train_size = int(len(train_set) * self.train_ratio)
        val_size = len(train_set) - train_size
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, val_size])

        return train_set, val_set

    def get_data_loader(self, batch_sizes, shuffle):
        datasets = self.load_data()
        data_loaders = []
        for dataset, batch_size in zip(datasets, batch_sizes):
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)
            data_loaders.append(data_loader)
        return data_loaders

    def train(self, epochs, save_every, batch_sizes, shuffle):
        criterion = nn.CrossEntropyLoss()
        train_loader, val_loader = self.get_data_loader(batch_sizes=batch_sizes,
                                                        shuffle=shuffle)

        for epoch in tqdm(range(epochs), desc='Epochs', total=epochs, leave=False,
                          unit='epoch'):
            for X, y in tqdm(train_loader, desc='Sample per batch',
                             total=batch_sizes[0], leave=False):
                self.model.train()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.train_history['train_loss'].append(loss.data.item())

            self.train_history['val_acc'].append(self.evaluate(val_loader=val_loader))

            if epoch % save_every == 0:
                self.save(epoch=epoch)

        return self.train_history

    def evaluate(self, val_loader):
        correct = 0
        total = 0
        for X, y in val_loader:
            total += 1
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.model(X)
            _, label = torch.max(y_hat, 1)
            correct += (label == y).sum().item()
        return correct / total

    def save(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f'checkpoint_{epoch}.pth')
        if self.scheduler:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, ckpt_path)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': None
            }, ckpt_path)
