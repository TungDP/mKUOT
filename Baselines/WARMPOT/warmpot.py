import argparse
import os
try:
    import wandb
except ImportError:
    wandb = None
import random
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import ot
import datasets as datasets
from utils import BalancedBatchSampler, CrossEntropyWeighted, Entropy
from datasets import default_partial as partial_dataset
from lr_schedule import schedule_dict
from network import ResNetFc_OfficeHome, ResNetFc_ImageNetCaltech
from sinkhorn import entropic_partial_wasserstein_logscale


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label
        ).item() / float(all_label.size()[0])

    return accuracy


def train(args):
    eta1 = args.eta1
    eta2 = args.eta2
    eta3 = args.eta3
    epsilon = args.epsilon
    beta = args.beta
    mass = args.mass
    print(f"WARMPOT algorithm")
    print(f"eta1 = {eta1}, eta2 = {eta2}, eta3 = {eta3}, epsilon = {epsilon}")
    print(f"beta = {beta}, mass = {mass}, seed = {args.seed}")
    log_str = "-" * 50 + "\n\n"
    print(log_str)

    # prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dataset = datasets.__dict__[args.dset]
    p_dataset = partial_dataset(dataset, args.tcls)
    dsets["source"] = dataset(root=args.dset_path, task=args.s_name, download=False, transform=image_train())
    dsets["target"] = p_dataset(root=args.dset_path, task=args.t_name, download=False, transform=image_train())
    dsets["test"] = p_dataset(root=args.dset_path, task=args.t_name, download=False, transform=image_test())

    dsets["source_val"] = p_dataset(root=args.dset_path, task=args.s_name, download=False, transform=image_test())

    dset_loaders = {}
    source_labels = torch.tensor(list(zip(*(dsets["source"].samples)))[1])

    if args.balanced_sampler:
        print("Using balanced sampler")
        train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = torch.utils.data.DataLoader(
            dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker
        )
    else:
        print("Not using balanced sampler")
        dset_loaders["source"] = DataLoader(
            dsets["source"], batch_size=train_bs, shuffle=True, 
            num_workers=args.worker, drop_last=True
        )

    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=True
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker
    )
    dset_loaders["source_val"] = DataLoader(
        dsets["source_val"], batch_size=test_bs, shuffle=False, num_workers=args.worker
    )

    params = {
        "resnet_name": args.net,
        "bottleneck_dim": 256,
        "class_num": args.class_num,
    }
    if args.dset == "OfficeHome":
        base_network = ResNetFc_OfficeHome(**params)

        optimizer_config = {
            "type": torch.optim.SGD,
            "optim_params": {
                "lr": args.lr, "weight_decay": 5e-4, 
                "momentum": 0.9, "nesterov": True,
            },
            "lr_type": "inv",
            "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
        }

    elif args.dset == "ImageNetCaltech":
        base_network = ResNetFc_ImageNetCaltech(**params)

        optimizer_config = {
            "type": torch.optim.Adam,
            "optim_params": {
                'lr': args.lr, "weight_decay": 5e-4,
            },
            "lr_type": "inv",
            "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}
        }

    else:
        raise ValueError(f'Unknown backbone {args.net}')

    base_network = base_network.cuda()

    parameter_list = base_network.get_parameters()

    optimizer = optimizer_config["type"](
        parameter_list, **(optimizer_config["optim_params"])
    )

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = schedule_dict[optimizer_config["lr_type"]]

    my_CrossEntropy = CrossEntropyWeighted()

    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    best_acc = 0
    best_iter = 0

    if args.mass_increase_i <= 1:
        adap_limit = -1
    else:
        adap_limit = args.mass_increase_i

    transfer_loss = torch.tensor(0.0).cuda().requires_grad_(True)

    for i in range(args.max_iterations + 1):

        if (i % args.test_interval == 0) \
            or (i == args.max_iterations) or (transfer_loss.isnan()):
            # compute test accuracy
            print(f'Start testing at iteration {i}.....')
            base_network.train(False)
            
            # sacc = image_classification(dset_loaders["source_val"], base_network)
            # sacc = np.round(sacc * 100, 2)
            # print(f'Source acc = {sacc}')
            # if args.use_wandb:
            #     wandb.log({f"src_acc": sacc})

            acc = image_classification(dset_loaders["test"], base_network)
            acc = np.round(acc * 100, 2)
            if acc > best_acc:
                best_acc = acc
                best_iter = i
                best_model = base_network.state_dict()

            print(f'Target acc = {acc}, best acc = {best_acc}')

            if args.use_wandb and wandb is not None:
                wandb.log({"tar_acc": acc})
                wandb.log({"best_tar_acc": best_acc})
            
            if transfer_loss.isnan():
                print("NaN loss, break training")
                return best_acc

        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        try:
            xs, ys, ids = next(iter_source)
            xt, _, _ = next(iter_target)
        except StopIteration:
            iter_source = iter(dset_loaders["source"])
            iter_target = iter(dset_loaders["target"])
            xs, ys, ids = next(iter_source)
            xt, _, _ = next(iter_target)
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        g_xs, f_g_xs = base_network(xs)
        g_xt, f_g_xt = base_network(xt)

        M_embed = torch.cdist(g_xs, g_xt)
        ys_one = F.one_hot(ys, num_classes=args.class_num).float().to(g_xs.device)
        log_pred_xt = torch.log_softmax(f_g_xt, dim=1)
        M_sce = -torch.mm(ys_one, log_pred_xt.T)
        M = eta1 * M_embed + eta2 * M_sce  # Ground cost
        # OT computation
        if i <= adap_limit:
            adap_mass = i * mass / adap_limit + 0.01
        else:
            adap_mass = mass

        a, b = ot.unif(g_xs.size()[0])/beta, ot.unif(g_xt.size()[0])
        M_cpu = M.detach().cpu().numpy()

        pi = entropic_partial_wasserstein_logscale(
            a, b, M_cpu, m=adap_mass, reg=epsilon
        )
        pi = torch.from_numpy(pi).float().cuda()
        transfer_loss = eta3 * torch.sum(pi * M)
        weights_p = torch.sum(pi, dim=1)
        classifier_loss = my_CrossEntropy(f_g_xs, ys_one, weights_p)

        target_ent = args.entropy * Entropy(f_g_xt).mean()

        if i % 100 == 0:
            print(f"\nstep = {i}, ",
                f"classifier = {classifier_loss.item():.2f}, ",
                f"transfer loss = {transfer_loss.item():.2f}, ",
                f"entropy loss = {target_ent.item():.2f}, ",
                f"sum(pi) = {torch.sum(pi).item():.2f}, ",
                f"alpha = {adap_mass:.3f}, M norm = {torch.norm(M)}\n")

        total_loss = classifier_loss + transfer_loss + target_ent
        total_loss.backward()

        optimizer.step()

    log_str = f"Acc: {best_acc}\n" 
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WARMPOT-PDA")

    parser.add_argument("--balanced_sampler", type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=0)

    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--output", type=str, default="run")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument(
        "--max_iterations", type=int, default=5000, help="max iterations"
    )
    parser.add_argument("--batch_size", type=int, default=65, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--net", type=str, default="ResNet50", choices=["ResNet50", "VGG16"]
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    parser.add_argument("--dset", type=str, default="office_home")
    parser.add_argument(
        "--test_interval", type=int, default=500, 
        help="interval of two continuous test phase"
    )
    parser.add_argument('--entropy', type=float, default=0.0, help='entropy weight')

    # WARMPOT parameters
    parser.add_argument(
        "--eta1", type=float, default=0.003, help="weight of embedding loss"
    )
    parser.add_argument(
        "--eta2", type=float, default=0.75, help="weight of transportation loss"
    )
    parser.add_argument(
        "--eta3", type=float, default=10.0, help="weight of transfer loss"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="OT regularization coefficient"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, 
        help="scales the source distribution by 1/beta"
    )
    parser.add_argument(
        "--mass", type=float, default=0.65, help="mass transported (alpha in the bound)"
    )
    parser.add_argument(
        "--mass_increase_i", type=int, default=2500, 
        help="number of iterations for which alpha is linearly increased before being fixed"
    )


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # import pdb; pdb.set_trace()
    if args.use_wandb and wandb is not None:
        wandb.init()

    if args.dset == "OfficeHome":
        names = ['Ar', 'Cl', 'Pr', 'Rw']
        args.s_name = names[args.s]
        args.t_name = names[args.t]

        args.tcls = 25
        args.class_num = 65

    if args.dset == "ImageNetCaltech":
        names = ['I', 'C']
        args.s_name = names[args.s]
        args.t_name = names[args.t]

        args.tcls = 0
        args.balanced_sampler = 0
        args.class_num = 1000
        if args.s == 1:
            args.class_num = 256


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = "./data/"
    args.dset_path = os.path.join(data_folder, args.dset)

    train(args)
