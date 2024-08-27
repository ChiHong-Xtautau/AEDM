import torch.nn.functional as F
import torchvision.transforms.functional as Ft

from src.fedrated_learning import *


class FLStealing(EasyFL):

    def __init__(self, args):
        super(FLStealing, self).__init__(args)

        self.diffusion = None
        self.target_net = None

        self.transform = torchvision.transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])

    def set_all_nets(self, target_path, pretrained_diff=None, path=None):
        self.set_client_models(path)
        self.set_diffusion(pretrained_diff)
        self.set_target(target_path)

    def set_target(self, target_path):
        if self.args.dataset == 'cifar100':
            net_arch = ResNet34_Cifar100
        else:
            # net_arch = ResNet34
            net_arch = ResNet18
        self.target_net = net_arch().to(self.device)

        state_dict = torch.load(target_path, map_location=self.device)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()

        acc = comm.accuracy(self.target_net, self.test_loader, self.device)
        logging.info('Accuracy of the target model: %.2f %%' % acc)

    def set_diffusion(self, pretrained_diff):
        self.diffusion = pretrained_diff

    def local_update(self, idx, global_round):
        local_net = self.clients[idx]
        local_net.train()

        if self.args.if_query:
            self.local_query(substitute=local_net, idx=idx, global_round=global_round)
        else:
            self.local_train(substitute=local_net, idx=idx, global_round=global_round)

    def local_train(self, substitute, idx, global_round):
        optimizer = torch.optim.RMSprop(substitute.parameters(), lr=0.0001)

        if self.args.substitute_loss == 'l1':
            criterion = self.l1_loss
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()

        inputs, target_outputs = self.load_batch(idx, global_round % self.args.max_batch_idx)

        outputs = substitute(inputs)
        loss = criterion(outputs, target_outputs)

        loss.backward()
        optimizer.step()

    def local_query(self, substitute, idx, global_round):
        optimizer = torch.optim.RMSprop(substitute.parameters(), lr=0.0001)

        if self.args.substitute_loss == 'l1':
            criterion = self.l1_loss
        else:
            criterion = nn.CrossEntropyLoss()

        noise = self.optimized_noise(substitute)
        substitute.train()
        optimizer.zero_grad()

        x_query = self.diffusion.sample(num_img=self.args.batch_size, input_noise=noise)

        inputs, target_outputs = self.get_query_batch(x_query)
        self.save_queries(inputs, target_outputs, idx, global_round)

        # training the substitute model
        outputs = substitute(inputs)
        loss = criterion(outputs, target_outputs)

        loss.backward()
        optimizer.step()

    def optimized_noise(self, substitute):
        substitute.eval()

        noise = torch.randn(self.args.batch_size, 3, 32, 32).to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([noise], lr=0.01)
        for i in range(5):
            optimizer.zero_grad()

            x = self.diffusion.sample(num_img=self.args.batch_size, input_noise=noise)
            s_out = substitute(x)
            s_out = F.softmax(s_out, dim=1)
            loss = self.negative_entropy(s_out)
            loss.backward()
            optimizer.step()
        return noise.requires_grad_(False)

    def get_query_batch(self, x_query):
        with torch.no_grad():
            target_outputs = self.target_net(x_query)
            target_outputs = F.softmax(target_outputs, dim=1)

        return x_query.to(self.device), target_outputs.to(self.device)

    def save_queries(self, x_query, target_outputs, idx, global_round):
        torch.save(x_query, self.args.save_path+'/queries/x_{}_{}.pth'.format(global_round, idx))
        torch.save(target_outputs, self.args.save_path + '/queries/y_{}_{}.pth'.format(global_round, idx))
        torchvision.utils.save_image(x_query, self.args.save_path+'/queries/visualize_{}_{}.png'.format(global_round, idx), nrow=10, padding=2)

    def load_batch(self, idx, global_round):
        x_query = torch.load(self.args.save_path+'/queries/x_{}_{}.pth'.format(global_round, idx))

        target_outputs = torch.load(self.args.save_path+'/queries/y_{}_{}.pth'.format(global_round, idx))

        if self.args.transform:
            x_query = self.transform(x_query)

        return x_query, target_outputs

    def l1_loss(self, s_out, t_out):
        t_logits = torch.log_(t_out)
        t_logits -= t_logits.mean(dim=1).view(-1, 1).detach()
        return F.l1_loss(s_out, t_logits)

    @staticmethod
    def negative_entropy(p):
        return torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=1))

