import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import logging
import os

from src.resnet import ResNet18, ResNet34, ResNet34_Cifar100
from src.vgg import VGG16
import src.common as comm


class EasyFL(object):

    def __init__(self, args):
        self.args = args

        self.global_net = None

        self.test_loader = None
        self.train_loader = None

        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.clients = []
        self.temp_loader = None

        self.set_logging()

    def set_logging(self):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        if not os.path.exists(self.args.save_path+'/queries'):
            os.makedirs(self.args.save_path+'/queries')

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler('{}/fl.log'.format(self.args.save_path)),
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
            test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                                    transform=transform_test)
            train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                                     transform=transform_train)
        elif self.args.dataset == 'svhn':
            test_set = torchvision.datasets.SVHN(root='./dataset', split='test', download=True,
                                                 transform=transform_test)
            train_set = torchvision.datasets.SVHN(root='./dataset', split='train', download=True,
                                                  transform=transform_train)
        else:
            test_set = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True,
                                                     transform=transform_test)
            train_set = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True,
                                                      transform=transform_train)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)

    def set_nets(self, path=None):
        self.set_client_models(path)

    def set_client_models(self, path=None):
        if self.args.dataset == 'cifar100':
            def net():
                return VGG16(num_classes=100)
            net_arch = net
        else:
            net_arch = VGG16

        self.global_net = net_arch().to(self.device)

        if path is not None:
            state_dict = torch.load(path, map_location=self.device)
            self.global_net.load_state_dict(state_dict)

        for k in range(self.args.num_clients):
            self.clients.append(net_arch().to(self.device))

        self.update_client_parameters(self.global_net.state_dict())

    def update_client_parameters(self, new_params):
        for k in range(self.args.num_clients):
            self.clients[k].load_state_dict(copy.deepcopy(new_params), strict=True)

    def aggregate(self):
        new_params = {}
        net_param_name_list = self.clients[0].state_dict().keys()
        num_clients = self.args.num_clients
        for name in net_param_name_list:
            new_params[name] = sum([self.clients[k].state_dict()[name].data for k in range(num_clients)]) / num_clients

        self.global_net.load_state_dict(copy.deepcopy(new_params), strict=True)
        return new_params

    def federated_train(self):
        acc = comm.accuracy(self.global_net, self.test_loader, self.device)
        logging.info('Accuracy of the global model: %.2f %%' % acc)

        for r in range(self.args.global_rounds):

            for k in range(self.args.num_clients):
                self.local_update(k, r)

            new_params = self.aggregate()
            self.update_client_parameters(new_params)

            if (r+1) % self.args.test_interval == 0:
                logging.info("round:{} / {}".format(r + 1, self.args.global_rounds))
                acc = comm.accuracy(self.global_net, self.test_loader, self.device)
                logging.info('Accuracy of the global model: %.2f %%' % acc)

        torch.save(self.global_net.state_dict(), '{}/final_global_model.pth'.format(self.args.save_path))

    def local_update(self, idx, global_round):
        local_net = self.clients[idx]
        local_net.train()

        optimizer = torch.optim.RMSprop(local_net.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        for c in range(self.args.num_local_steps):
            optimizer.zero_grad()

            inputs, labels = self.get_next_batch()

            outputs = local_net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    def get_next_batch(self):
        try:
            data, target = next(self.temp_loader)
        except StopIteration:
            self.temp_loader = iter(self.train_loader)
            data, target = next(self.temp_loader)
        return data.to(self.device), target.to(self.device)

