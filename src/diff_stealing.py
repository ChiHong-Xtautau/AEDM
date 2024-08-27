import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
from tqdm.auto import tqdm
import logging
import os
import random

from src.resnet import ResNet18, ResNet34, ResNet34_Cifar100, ResNet18_Cifar100
from src.vgg import VGG19, VGG16, VGG13, VGG11
import src.common as comm


class DiffStealing(object):

    def __init__(self, args):
        self.args = args

        self.test_loader = None
        self.train_loader = None

        self.target_net = None

        self.diff = None

        self.substitute = None

        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.set_logging()

        self.idx_query = 0

        self.transform = torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])

    def set_logging(self):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        if not os.path.exists(self.args.save_path+'/queries'):
            os.makedirs(self.args.save_path+'/queries')

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler('{}/{}_queries_{}.log'.format(
                    self.args.save_path, self.args.dataset, self.args.queries_per_stage*self.args.num_stages)),
                logging.StreamHandler()
            ]
        )
        logging.info(self.args)

    def load_data(self):
        transform_train = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.args.dataset == 'cifar10':
            test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
            train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
        elif self.args.dataset == 'svhn':
            test_set = torchvision.datasets.SVHN(root='./dataset', split='test', download=True, transform=transform_test)
            train_set = torchvision.datasets.SVHN(root='./dataset', split='train', download=True, transform=transform_train)
        else:
            test_set = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
            train_set = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)

    def set_all_nets(self, target_path, pretrained_diff, path=None):
        self.set_substitute(path)
        self.set_generators(pretrained_diff)
        self.set_target(target_path)

    def set_generators(self, pretrained_diff):
        self.diff = pretrained_diff

    def set_substitute(self, path=None):
        if self.args.substitute_net == 'ResNet18':
            if self.args.dataset == 'cifar100':
                net_arch = ResNet18_Cifar100
            else:
                net_arch = ResNet18
        elif self.args.substitute_net == 'VGG16':
            if self.args.dataset == 'cifar100':
                def net():
                    return VGG16(num_classes=100)
                net_arch = net
            else:
                net_arch = VGG16
        elif self.args.substitute_net == 'VGG13':
            if self.args.dataset == 'cifar100':
                def net():
                    return VGG13(num_classes=100)
                net_arch = net
            else:
                net_arch = VGG13
        elif self.args.substitute_net == 'VGG11':
            if self.args.dataset == 'cifar100':
                def net():
                    return VGG11(num_classes=100)
                net_arch = net
            else:
                net_arch = VGG11
        else:
            if self.args.dataset == 'cifar100':
                def net():
                    return VGG19(num_classes=100)
                net_arch = net
            else:
                net_arch = VGG19

        self.substitute = net_arch().to(self.device)

        if path is not None:
            state_dict = torch.load(path, map_location=self.device)
            self.substitute.load_state_dict(state_dict)
        self.substitute.train()

    def set_target(self, target_path=None):
        if self.args.dataset == 'cifar100':
            net_arch = ResNet34_Cifar100
        else:
            # net_arch = ResNet34
            net_arch = ResNet18

        self.target_net = net_arch().to(self.device)
        # self.target_net = nn.DataParallel(self.target_net)

        state_dict = torch.load(target_path, map_location=self.device)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()

        acc = comm.accuracy(self.target_net, self.test_loader, self.device)
        logging.info('Accuracy of the target model: %.2f %%' % acc)

    def show_substitute_acc(self):
        acc = comm.accuracy(self.substitute, self.test_loader, self.device)
        logging.info('Accuracy of the substitute model: %.2f %%' % acc)

    def steal(self):
        self.show_substitute_acc()
        for i in range(self.args.num_stages):
            logging.info('stage: {} / {}'.format(i+1, self.args.num_stages))
            self.query()
            # self.idx_query = 50
            self.train_substitute()

    def train_substitute(self):
        logging.info('Training:')

        optimizer = torch.optim.RMSprop(self.substitute.parameters(), lr=0.0001)
        # optimizer = torch.optim.Adam(self.substitute.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=0.2)

        if self.args.substitute_loss == 'l1':
            criterion = self.l1_loss
        else:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.epochs_per_stage):
            logging.info('epoch: {} / {}'.format(epoch+1, self.args.epochs_per_stage))
            num_batch = self.idx_query

            batch_order = list(range(num_batch))
            random.shuffle(batch_order)
            for i in tqdm(range(num_batch), desc="batch idx", total=num_batch):
                optimizer.zero_grad()

                # inputs, target_outputs = self.load_batch(i)
                inputs, target_outputs = self.load_batch(batch_order[i])

                outputs = self.substitute(inputs)
                loss = criterion(outputs, target_outputs)

                loss.backward()
                optimizer.step()

            scheduler.step()
            self.show_substitute_acc()
        torch.save(self.substitute.state_dict(), '{}/substitute.pth'.format(self.args.save_path))

    def optimized_noise(self):
        self.substitute.eval()

        noise = torch.randn(self.args.batch_size, 3, 32, 32).to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([noise], lr=0.01)
        for i in range(5):
            optimizer.zero_grad()

            x = self.diff.sample(num_img=self.args.batch_size, input_noise=noise)
            s_out = self.substitute(x)
            s_out = F.softmax(s_out, dim=1)
            loss = self.negative_entropy(s_out)
            loss.backward()
            optimizer.step()
        return noise.requires_grad_(False)

    def query(self):
        logging.info('Querying:')
        optimizer = torch.optim.RMSprop(self.substitute.parameters(), lr=0.0001)

        if self.args.substitute_loss == 'l1':
            criterion = self.l1_loss
        else:
            criterion = nn.CrossEntropyLoss()

        num_batch = int(self.args.queries_per_stage / self.args.batch_size)
        for i in tqdm(range(num_batch), desc='batches', total=num_batch):
            noise = self.optimized_noise()

            self.substitute.train()
            optimizer.zero_grad()

            x_query = self.diff.sample(num_img=self.args.batch_size, input_noise=noise)

            inputs, target_outputs = self.get_query_batch(x_query)
            self.save_queries(inputs, target_outputs)

            # training the substitute model
            outputs = self.substitute(inputs)
            loss = criterion(outputs, target_outputs)

            loss.backward()
            optimizer.step()

        self.show_substitute_acc()

    def save_queries(self, x_query, target_outputs):
        torch.save(x_query, self.args.save_path+'/queries/x_{}.pth'.format(self.idx_query))
        torch.save(target_outputs, self.args.save_path + '/queries/y_{}.pth'.format(self.idx_query))
        torchvision.utils.save_image(x_query, self.args.save_path+'/queries/visualize_{}.png'.format(self.idx_query), nrow=10, padding=2)
        self.idx_query += 1

    def load_batch(self, idx):
        x_query = torch.load(self.args.save_path+'/queries/x_{}.pth'.format(idx))
        x_query = x_query.to(self.device)

        target_outputs = torch.load(self.args.save_path+'/queries/y_{}.pth'.format(idx))

        if self.args.transform:
            x_query = self.transform(x_query)

        return x_query, target_outputs

    def get_query_batch(self, x_query):
        with torch.no_grad():
            target_outputs = self.target_net(x_query)
            target_outputs = F.softmax(target_outputs, dim=1)

        return x_query.to(self.device), target_outputs.to(self.device)

    def l1_loss(self, s_out, t_out):
        t_logits = torch.log_(t_out)
        t_logits -= t_logits.mean(dim=1).view(-1, 1).detach()
        return F.l1_loss(s_out, t_logits)

    @staticmethod
    def negative_entropy(p):
        return torch.mean(torch.sum(p * torch.log(p+1e-8), dim=1))