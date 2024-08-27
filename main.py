import argparse
import math

from src.diff_stealing import DiffStealing
from src.fl_stealing import FLStealing
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.diffusion_utils import DiffusionUtils


def build_diffusion(image_size, objective="pred_v", timesteps=1024, sampling_timesteps=None,
                    teacher=None, using_ddim=False):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,  # 32, 64, 128
        timesteps=timesteps,  # number of steps
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        teacher=teacher,
        using_ddim=using_ddim,
    ).cuda()
    return diffusion


def load_pretrained_gen():
    diff_util = DiffusionUtils(
        build_diffusion(image_size=32, timesteps=8, sampling_timesteps=8, objective="pred_v", using_ddim=True))
    d_dir = 'trained_models/diffusion_models/diffusion_Imagenet_32x32_8.pth'
    diff_util.load_trained_model(d_dir)
    return diff_util


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    args.add_argument('--dataset', type=str, default='cifar10', help='cifar10, svhn, cifar100')

    args.add_argument('--batch_size', type=int, default=10)
    args.add_argument('--transform', default=True, action='store_true')
    args.add_argument('--z_dim', type=int, default=128)
    args.add_argument('--substitute_net', type=str, default='ResNet18', help="ResNet18, VGG13, VGG16, VGG19")
    args.add_argument('--substitute_loss', type=str, default='l1', help='l1, ce')
    args.add_argument('--queries_per_stage', type=int, default=20000)
    args.add_argument('--epochs_per_stage', type=int, default=100)
    args.add_argument('--milestones', nargs='+', default=[60, 80], type=float)
    args.add_argument('--num_stages', type=int, default=1)

    args.add_argument('--save_path', type=str, default='res/cifar100_resnet34_resnet18')

    args = args.parse_args()
    return args


def run_diff_stealing(args):
    diff_stealing = DiffStealing(args)
    diff_stealing.load_data()
    diff_stealing.set_all_nets(target_path='trained_models/target_models/Cifar10_resnet18_nonormalize.pth',
                               pretrained_diff=load_pretrained_gen(), path=None)
    diff_stealing.steal()


def get_fl_args():
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    args.add_argument('--dataset', type=str, default='cifar10', help='cifar10, svhn, cifar100')

    args.add_argument('--num_clients', type=int, default=20)
    args.add_argument('--global_rounds', type=int, default=10000)
    args.add_argument('--test_interval', type=int, default=10)
    args.add_argument('--if_query', default=True, action='store_true')
    args.add_argument('--max_batch_idx', type=int, default=99)

    args.add_argument('--batch_size', type=int, default=50)
    args.add_argument('--transform', default=True, action='store_true')
    args.add_argument('--z_dim', type=int, default=128)
    args.add_argument('--substitute_net', type=str, default='VGG16', help="ResNet18, VGG13, VGG16, VGG19")
    args.add_argument('--substitute_loss', type=str, default='l1', help='l1, ce')
    args.add_argument('--queries_per_stage', type=int, default=100000)
    args.add_argument('--epochs_per_stage', type=int, default=200)
    args.add_argument('--milestones', nargs='+', default=[120, 160, 180], type=float)
    args.add_argument('--num_stages', type=int, default=1)

    args.add_argument('--save_path', type=str, default='res/fl_cifar10_resnet34_vgg16')

    args = args.parse_args()
    return args


def run_fl_stealing(args):
    fl_stealing = FLStealing(args)
    fl_stealing.load_data()
    fl_stealing.set_all_nets(target_path='trained_models/target_models/Cifar10_resnet18_nonormalize.pth',
                               pretrained_diff=load_pretrained_gen(), path=None)
    fl_stealing.federated_train()


if __name__ == '__main__':
    args = get_args()
    run_diff_stealing(args)

    # args = get_fl_args()
    # run_fl_stealing(args)
