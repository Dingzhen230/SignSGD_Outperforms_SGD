import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging

from resnet import ResNet20_CIFAR
from gradient_utils import sample_cifar_gradient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class NetworkTrainer:
    def __init__(self, optim_name, num_repeats, lr, wd, device, local_rank, **kwargs):
        self.optim_name = optim_name
        self.num_repeats = num_repeats
        self.lr = lr
        self.wd = wd
        self.device = device
        self.local_rank = local_rank
        self.is_master = (local_rank == 0)
        
        if optim_name == 'adam':
            self.betas = (kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999))
            self.epsilon = kwargs.get("epsilon", 1e-8)
            self.momentum = self.betas[0]
        else:
            self.momentum = kwargs.get("momentum", 0.9)
            
        self.batch_size = 128
        self.epochs = 160
        
        if self.is_master:
            logging.info(f"Initialized {optim_name.upper()} Trainer | LR: {lr} | WD: {wd} | Momentum: {self.momentum}")

        self.get_data()

    def get_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        self.train_sampler = DistributedSampler(trainset)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=4)

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def define_network(self):
        self.net = ResNet20_CIFAR(num_classes=10).to(self.device)
        self.net = DDP(self.net, device_ids=[self.local_rank])

    def get_optimizer(self):
        if self.optim_name == 'adam':
            return optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.betas, eps=self.epsilon)
        elif self.optim_name == 'signum':
            if self.is_master: logging.warning("Signum placeholder: Using SGD. Implement exact SignSGD updates if needed.")
            return optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        else: # sgd
            return optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

    def evaluate(self):
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    def train_repeatedly(self):
        for repeat in range(self.num_repeats):
            if self.is_master: logging.info(f"\n{'='*40}\nStarting Repeat {repeat+1}/{self.num_repeats}\n{'='*40}")
            self.define_network()
            self.train(repeat)

    def train(self, repeat):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer()
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

        history = {
            'train_acc': [], 'test_acc': [], 'loss': [],
            'd': [], 'l1_sq': [], 'l2_sq': [], 'var': [], 's1_sig': [], 's2_sig': [],
            'mean_density': [], 'phi': []
        }

        if self.is_master:
            os.makedirs("./samples", exist_ok=True)
            os.makedirs("./results", exist_ok=True)

        for epoch in range(self.epochs):
            self.train_sampler.set_epoch(epoch)

            if self.is_master: logging.info(f"Epoch {epoch} | Computing DDP Gradient Statistics...")
            w_mean, w_var, grad_samples = sample_cifar_gradient(self.train_loader, self.net, criterion, self.device)

            d = w_mean.numel()
            w_sig = torch.sqrt(w_var)
            
            l1_sq = torch.norm(w_mean, p=1).item() ** 2
            l2_sq = torch.norm(w_mean, p=2).item() ** 2
            total_var = torch.sum(w_var).item()
            
            s1_sig = torch.norm(w_sig, p=1).item() ** 2
            s2_sig = torch.norm(w_sig, p=2).item() ** 2

            mean_density = l1_sq / (d * l2_sq) if l2_sq > 0 else 0.0
            phi = s1_sig / (d * s2_sig) if s2_sig > 0 else 0.0

            if self.is_master:
                logging.info(f"[Stats] Dim: {d} | Var: {total_var:.4f} | Sparsity (\phi): {phi:.4f}")
                torch.save(grad_samples, f'./samples/samples_{self.optim_name}_rep{repeat}_epoch{epoch}.pt')

            for key, val in zip(['d', 'l1_sq', 'l2_sq', 'var', 's1_sig', 's2_sig', 'mean_density', 'phi'], 
                                [d, l1_sq, l2_sq, total_var, s1_sig, s2_sig, mean_density, phi]):
                history[key].append(val)

            self.net.train()
            train_loss, correct, total = 0.0, 0, 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total
            test_acc = self.evaluate()
            epoch_loss = train_loss / len(self.train_loader)
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(epoch_loss)

            if self.is_master:
                logging.info(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Loss: {epoch_loss:.4f}\n")

        if self.is_master:
            torch.save(history, f'./results/history_{self.optim_name}_rep{repeat}.pt')

            csv_filename = f'./results/history_{self.optim_name}_rep{repeat}.csv'
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_acc', 'dim_d', 'total_variance', 'mean_density', 'sparsity_phi'])
                
                for epoch in range(self.epochs):
                    writer.writerow([
                        epoch, history['loss'][epoch], history['train_acc'][epoch], history['test_acc'][epoch],
                        history['d'][epoch], history['var'][epoch], history['mean_density'][epoch], history['phi'][epoch]
                    ])
                    
            logging.info(f"✅ Training repeat {repeat} completed. CSV saved to: {csv_filename}")