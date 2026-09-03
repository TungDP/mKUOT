import argparse
import copy
import os
import random
import sys
import time
import ot
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from utils import datasets
from utils.meter import AverageMeter, ProgressMeter
from utils.sampler import BalancedBatchSampler
from utils.logger import TextLogger
from utils import h_score, entropy_loss, get_prototypes
from modules.resnet import Res50
from moco.moco import train_moco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

# VisDA-2017 class order (label 0..11). Abbreviations match the standard
# per-class result table (e.g. Table 7 of the m-POT paper) used to report VisDA.
VISDA_CLASSES = ['plane', 'bcycl', 'bus', 'car', 'house', 'knife',
                 'mcycl', 'person', 'plant', 'sktbrd', 'train', 'truck']
VISDA_HEADERS = ['Run', 'Method', 'All'] + VISDA_CLASSES + ['Avg']
# fixed column widths so rows stay aligned across appends (Run, Method, then
# the 14 numeric columns: All + 12 classes + Avg).
VISDA_COL_W = [4, 10, 7] + [7] * len(VISDA_CLASSES) + [7]


def _visda_line(cells):
    # Method (index 1) is left-aligned; everything else right-aligned, booktabs style.
    out = []
    for i, (c, w) in enumerate(zip(cells, VISDA_COL_W)):
        out.append(str(c).ljust(w) if i == 1 else str(c).rjust(w))
    return '  '.join(out)


def visda_header_lines():
    header = _visda_line(VISDA_HEADERS)
    return header, '-' * len(header)


def visda_row(run, method, per_class_acc, all_acc, avg_acc):
    cells = [run, method, '{:.2f}'.format(all_acc * 100)] \
        + ['{:.2f}'.format(a * 100) for a in per_class_acc] \
        + ['{:.2f}'.format(avg_acc * 100)]
    return _visda_line(cells)


def print_visda_table(run, per_class_acc, all_acc, avg_acc, method='PPOT',
                      out_file=None, to_console=True, header=True):
    """Render a VisDA-2017 result row (Run | Method | All | 12 classes | Avg,
       values in %). Prints to console when to_console is True, and/or appends
       to out_file. `run` may be an int run id or a tag like 'e3'/'best'."""
    h, rule = visda_header_lines()
    row = visda_row(run, method, per_class_acc, all_acc, avg_acc)
    text = '\n'.join(([h, rule] if header else []) + [row])
    if to_console:
        print(text)
    if out_file:
        with open(out_file, 'a') as f:
            f.write(text + '\n')


def next_visda_run(task_file):
    """Next Run number for a per-task results file (max existing Run id + 1)."""
    if not os.path.exists(task_file):
        return 1
    mx = 0
    with open(task_file) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0].isdigit():
                mx = max(mx, int(tok[0]))
    return mx + 1


def append_visda_result(run, per_class_acc, all_acc, avg_acc, task_file, method='PPOT'):
    """Append one best-result row to the per-task results .txt (one row per run,
       Table-7 layout). Writes the header + rule once when the file is created."""
    new_file = not os.path.exists(task_file)
    h, rule = visda_header_lines()
    row = visda_row(run, method, per_class_acc, all_acc, avg_acc)
    with open(task_file, 'a') as f:
        if new_file:
            f.write(h + '\n' + rule + '\n')
        f.write(row + '\n')


def main(args: argparse.Namespace):
    # create logger
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    filename = os.path.join('log/', "{}2{}-{}.txt".format(args.source, args.target, now))
    logger = TextLogger(filename)
    sys.stdout = logger
    sys.stderr = logger
    # share the run log path with the MoCo subprocess so its (spawned) master
    # worker can append its progress here too (otherwise the file looks frozen
    # during MoCo pretraining).
    args.log_file = filename

    # one results file per task (Table-7 layout): each run appends one best row.
    args.summary_file = os.path.join('log/', "{}_results.txt".format(args.task))
    args.run_no = next_visda_run(args.summary_file)

    if args.task == 'office31':
        args.common_class = 10
        args.source_private_class = 10
        args.target_private_class = 11
        args.moco_k = 1024
    if args.task == 'VisDA2017':
        args.common_class = 6
        args.source_private_class = 3
        args.target_private_class = 3
        args.moco_k = 65536
    if args.task == 'officehome':
        args.common_class = 10
        args.source_private_class = 5
        args.target_private_class = 50
        args.moco_k = 3072
    # ── Closed-set override (for fair comparison with closed-set DA methods
    # such as m-KPOT) — all classes are shared, no unknowns. ─────────────
    if getattr(args, 'closed_set', False):
        if args.task == 'officehome':
            args.common_class = 65
            args.source_private_class = 0
            args.target_private_class = 0
        elif args.task == 'office31':
            args.common_class = 31
            args.source_private_class = 0
            args.target_private_class = 0
        elif args.task == 'VisDA2017':
            args.common_class = 12
            args.source_private_class = 0
            args.target_private_class = 0
        print(f"[CLOSED-SET MODE] common={args.common_class}, "
              f"src_private={args.source_private_class}, "
              f"tgt_private={args.target_private_class}")
    if args.task == 'DomainNet':
        args.batch_size = 256
        args.common_class = 150
        args.source_private_class = 50
        args.target_private_class = 145
        args.moco_k = 65536
    print(args)

    # create seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.enabled = True

    # create transform
    train_transform = transform.Compose([
        transform.Resize(256),
        transform.RandomResizedCrop(224),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # create data set
    source_dataset = datasets.Datasets(args, source=True, transform=train_transform)
    if args.balanced:
        source_label = source_dataset.create_label_set()
        train_batch_sampler = BalancedBatchSampler(source_label, batch_size=args.batch_size)
        source_loader = DataLoader(source_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers)
    else:
        source_loader = DataLoader(dataset=source_dataset, batch_size=args.source_batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)

    target_dataset = datasets.Datasets(args, source=False, transform=train_transform)
    target_loader = DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    val_dataset = datasets.Datasets(args, source=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    num_class = args.common_class + args.source_private_class
    if args.checkpoint:
        model = Res50(num_class=num_class, checkpoint=args.checkpoint).to(device)
        print("use checkpoint")
    elif args.no_ssl:
        model = Res50(num_class=num_class).to(device)
    else:
        train_moco(args)
        model = Res50(num_class=num_class,
                      checkpoint='checkpoint/{}2{}_{:04d}.pth.tar'.format(args.source, args.target, args.moco_epochs)
                      ).to(device)

    # create optimizer
    all_parameters = model.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=10 * args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # train source domain 1000 iterations to fine-tune the model
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    progress = ProgressMeter(
        args.pre_step,
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(-1))

    model.train()
    if args.new_opt:
        # use an independent optimizer to fine-tune (optional)
        new_optimizer = SGD(all_parameters, args.pre_lr, momentum=args.momentum, weight_decay=args.weight_decay,
                            nesterov=True)
        new_lr_scheduler = LambdaLR(new_optimizer,
                                    lambda x: args.pre_lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    source_iter = iter(source_loader)
    end = time.time()
    for step in range(args.pre_step):
        try:
            source_data = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_data = next(source_iter)
        s_img, s_label = source_data
        s_img = s_img.to(device)
        s_label = s_label.to(device)
        data_time.update(time.time() - end)
        s_prediction, s_feature = model(s_img)

        loss = F.cross_entropy(s_prediction, s_label)

        if args.new_opt:
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()
            new_lr_scheduler.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        with torch.no_grad():
            pred = torch.argmax(s_prediction, dim=1)
            correct = pred.eq(s_label).sum().item()
            total_correct = correct
            total_samples = s_label.size(0)
            acc = total_correct / total_samples
            top1.update(acc, s_img.size(0))
            losses.update(loss, s_img.size(0))
            if (step + 1) % 100 == 0:
                progress.display(step + 1)

    # start training
    best_h_score = 0.
    best_unknown_acc = 0.
    best_known_acc = 0.
    best_avg = 0.
    best_all = 0.
    best_per_class = None

    validate(val_loader, model, epoch=-1)

    # init source prototypes, alpha, beta and class weight
    s_feat, s_label = get_features(source_loader, model)
    s_protos = get_prototypes(s_feat, s_label, args)
    alpha = update_alpha(target_loader, model)
    beta = alpha
    class_weight = torch.ones(num_class)

    for epoch in range(args.epochs):
        # train one epoch
        class_weight, beta = train(source_loader, target_loader, s_protos, class_weight, model, optimizer, lr_scheduler,
                                   alpha, beta, epoch)

        # update source prototypes and alpha
        s_feat, s_label = get_features(source_loader, model)
        s_protos = get_prototypes(s_feat, s_label, args)
        alpha = update_alpha(target_loader, model)

        # evaluate on validation set
        h_scores, k_acc, u_acc, all_acc, avg_acc, per_class_acc = validate(val_loader, model, epoch)

        # remember best acc@1 and save checkpoint
        if h_scores * 100 > best_h_score:
            best_h_score = h_scores * 100
            best_unknown_acc = u_acc * 100
            best_known_acc = k_acc * 100

        # for closed-set VisDA track the best mean-per-class accuracy ("Avg")
        if avg_acc * 100 > best_avg:
            best_avg = avg_acc * 100
            best_all = all_acc * 100
            best_per_class = per_class_acc

    print("best H-score = {:3.1f}".format(best_h_score))
    print("best unknown accuracy = {:3.1f}".format(best_unknown_acc))
    print("best known accuracy = {:3.1f}".format(best_known_acc))

    # append this run's best result (one row) to the per-task results file
    if args.task == 'VisDA2017' and getattr(args, 'closed_set', False) and best_per_class is not None:
        print('\n===== BEST VisDA-2017 result (Run {}) ====='.format(args.run_no))
        print_visda_table(args.run_no, best_per_class, best_all / 100, best_avg / 100,
                          method=args.method_name)
        append_visda_result(args.run_no, best_per_class, best_all / 100, best_avg / 100,
                            args.summary_file, method=args.method_name)
        print('per-task results written to: {} (Run {})'.format(args.summary_file, args.run_no))


def train(source_loader: DataLoader, target_loader: DataLoader, source_prototype: torch.Tensor,
          class_weight: torch.Tensor, model: Res50, optimizer: SGD, lr_scheduler: LambdaLR,
          alpha: np.ndarray, beta: np.ndarray, epoch: int) -> [torch.Tensor, np.ndarray]:

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    cls_losses = AverageMeter('C_Loss', ':.2e')
    ot_losses = AverageMeter('OT_Loss', ':.2e')
    progress = ProgressMeter(
        args.step,
        [batch_time, cls_losses, ot_losses],
        prefix="Epoch: [{}]".format(epoch))
    num_classes = args.common_class + args.source_private_class
    model.train()

    source_data_iter = iter(source_loader)
    target_data_iter = iter(target_loader)
    end = time.time()
    for step in range(args.step):
        try:
            source_data = next(source_data_iter)
        except StopIteration:
            source_data_iter = iter(source_loader)
            source_data = next(source_data_iter)
        try:
            target_data = next(target_data_iter)
        except StopIteration:
            target_data_iter = iter(target_loader)
            target_data = next(target_data_iter)

        s_img, s_label = source_data
        t_img, _ = target_data
        s_img, t_img = s_img.to(device), t_img.to(device)
        s_label = s_label.to(device)
        class_weight = class_weight.to(device)
        data_time.update(time.time() - end)
        s_prediction, s_feature = model(s_img)
        s_feature = F.normalize(s_feature, p=2, dim=-1)
        _, t_feature = model(t_img)
        
        # freeze head parameters
        head = copy.deepcopy(model.head)
        for params in head.parameters():
            params.requires_grad = False
        t_prediction = F.softmax(head(t_feature), dim=1)
        conf, pred = t_prediction.max(dim=1)
        t_feature = F.normalize(t_feature, p=2, dim=-1)
        batch_size = t_feature.shape[0]

        # update alpha by moving average
        alpha = (1 - args.alpha) * alpha + args.alpha * (conf >= args.tau1).sum().item() / conf.size(0)

        # get alpha / beta
        match = alpha / beta

        # update source prototype by moving average
        source_prototype = source_prototype.data.to(device)
        batch_source_prototype = torch.zeros_like(source_prototype).to(device)
        for i in range(num_classes):
            if (s_label == i).sum().item() > 0:
                batch_source_prototype[i] = (s_feature[s_label == i].mean(dim=0))
            else:
                batch_source_prototype[i] = (source_prototype[i])
        source_prototype = (1 - args.tau) * source_prototype + args.tau * batch_source_prototype
        source_prototype = F.normalize(source_prototype, p=2, dim=-1)

        # get ot loss
        a, b = match * ot.unif(num_classes), ot.unif(batch_size)
        m = torch.cdist(source_prototype, t_feature) ** 2
        m_max = m.max().detach()
        m = m / m_max
        pi, log = ot.partial.entropic_partial_wasserstein(a, b, m.detach().cpu().numpy(), reg=args.reg, m=alpha,
                                                          stopThr=1e-10, log=True)
        pi = torch.from_numpy(pi).float().to(device)
        ot_loss = torch.sqrt(torch.sum(pi * m) * m_max)
        loss = args.ot * ot_loss

        # update class weight and target weight by plan pi
        plan = pi * batch_size
        k = round(args.neg * batch_size)
        min_dist, _ = torch.min(m, dim=0)
        _, indicate = min_dist.topk(k=k, dim=0)
        batch_class_weight = torch.tensor([plan[i, :].sum() for i in range(num_classes)]).to(device)
        class_weight = args.tau * batch_class_weight + (1 - args.tau) * class_weight
        class_weight = class_weight * num_classes / class_weight.sum()
        k_weight = torch.tensor([plan[:, i].sum() for i in range(batch_size)]).to(device)
        k_weight /= alpha
        u_weight = torch.zeros(batch_size).to(device)
        u_weight[indicate] = 1 - k_weight[indicate]

        # update beta
        beta = args.beta * (class_weight > args.tau2).sum().item() / num_classes + (1 - args.beta) * beta

        # get classification loss
        cls_loss = F.cross_entropy(s_prediction, s_label, weight=class_weight.float())
        loss += cls_loss

        # get entropy loss
        p_ent_loss = args.p_entropy * entropy_loss(t_prediction, k_weight)
        n_ent_loss = args.n_entropy * entropy_loss(t_prediction, u_weight)
        ent_loss = p_ent_loss - n_ent_loss
        loss += ent_loss

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        with torch.no_grad():

            cls_losses.update(cls_loss, batch_size)
            ot_losses.update(args.ot * ot_loss, batch_size)

            if (step + 1) % args.interval == 0:
                progress.display(step + 1)
    return class_weight, beta


def validate(val_loader: DataLoader, model: Res50, epoch: int) -> [np.ndarray]:
    data_time = AverageMeter('Time', ':6.3f')
    k_top1 = AverageMeter('Known', ':6.3f')
    u_top1 = AverageMeter('Unknown', ':6.3f')
    h_sco = AverageMeter('H-score', ':6.3f')
    progress1 = ProgressMeter(
        len(val_loader),
        [data_time, k_top1, u_top1, h_sco],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()
    total_correct = 0
    total_samples = 0
    total_unknown_correct = 0
    total_unknown_samples = 0
    unknown_label = args.common_class + args.source_private_class
    num_class = args.common_class + args.source_private_class
    # per-class accuracy accumulators (raw argmax, i.e. the closed-set / VisDA metric)
    per_class_correct = torch.zeros(num_class)
    per_class_total = torch.zeros(num_class)
    end = time.time()
    with torch.no_grad():
        for step, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            output, feature = model(image)
            softmax = nn.Softmax(dim=1)
            output = softmax(output)

            # discriminate unknown sample by confidence
            confidence, pred = output.max(dim=1)

            # per-class accuracy on the raw argmax prediction (before the unknown
            # relabel below) — this is the standard closed-set VisDA metric.
            for c in range(num_class):
                cmask = (label == c)
                n_c = cmask.sum().item()
                if n_c > 0:
                    per_class_total[c] += n_c
                    per_class_correct[c] += pred[cmask].eq(c).sum().item()

            pred[confidence < args.threshold] = unknown_label
            correct = pred.eq(label).sum().item()
            unknown_correct = pred[confidence < args.threshold].eq(label[confidence < args.threshold]).sum().item()
            unknown_sample = (label == unknown_label).sum().item()
            total_correct += correct
            total_unknown_correct += unknown_correct
            total_samples += label.shape[0]
            total_unknown_samples += unknown_sample

        # compute H-score and accuracy
        known_acc = (total_correct - total_unknown_correct) / (total_samples - total_unknown_samples)
        if total_unknown_samples > 0:
            unknown_acc = total_unknown_correct / total_unknown_samples
            h_scores = h_score(known_acc, unknown_acc)
        else:
            # closed-set: no unknown samples exist, H-score is not defined
            unknown_acc = 0.0
            h_scores = known_acc
        data_time.update(time.time() - end)
        k_top1.update(known_acc)
        u_top1.update(unknown_acc)
        h_sco.update(h_scores)
        progress1.display(len(val_loader))

        # per-class accuracy, overall accuracy ("All") and mean-per-class ("Avg")
        valid = per_class_total > 0
        per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
        all_acc = (per_class_correct.sum() / per_class_total.sum().clamp(min=1)).item()
        avg_acc = (per_class_correct[valid] / per_class_total[valid]).mean().item() if valid.any() else 0.0

        # show the current epoch's VisDA-2017 result row on the console for
        # live monitoring (the per-task file gets the best row at the end)
        if args.task == 'VisDA2017' and getattr(args, 'closed_set', False):
            print_visda_table('e{}'.format(epoch), per_class_acc, all_acc, avg_acc)
    return h_scores, known_acc, unknown_acc, all_acc, avg_acc, per_class_acc


def update_alpha(target_loader: DataLoader, model: Res50) -> np.ndarray:
    num_conf, num_sample = 0, 0
    model.eval()

    with torch.no_grad():
        for _, (img, _) in enumerate(target_loader):
            img = img.to(device)
            output, _ = model(img)
            output = F.softmax(output, dim=1)
            conf, _ = output.max(dim=1)
            num_conf += torch.sum(conf > args.tau1).item()
            num_sample += output.shape[0]

        alpha = num_conf / num_sample
        alpha = np.around(alpha, decimals=2)
    return alpha


def get_features(data_loader: DataLoader, model: Res50) -> [torch.Tensor, torch.Tensor]:
    feature_set = []
    label_set = []
    model.eval()
    with torch.no_grad():
        for _, (img, gt) in enumerate(data_loader):
            img = img.to(device)
            _, feature = model(img)
            feature_set.append(feature)
            label_set.append(gt)
        feature_set = torch.cat(feature_set, dim=0)
        feature_set = F.normalize(feature_set, p=2, dim=-1)
        label_set = torch.cat(label_set, dim=0)
    return feature_set, label_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPOT for Universal Domain Adaptation')
    parser.add_argument('--root', default='/path/to/your/dataset/',
                        help='root of data file')
    parser.add_argument('--task', default='officehome',
                        help='task name')
    parser.add_argument('-s', '--source', default='Art',
                        help='source domain')
    parser.add_argument('-t', '--target', default='Clipart',
                        help='target domain')
    parser.add_argument('--common-class', default=10, type=int,
                        help='number of common class')
    parser.add_argument('--source-private-class', default=5, type=int,
                        help='number of source private class')
    parser.add_argument('--target-private-class', default=50, type=int,
                        help='number of target private class')
    parser.add_argument('-b', '--batch-size', default=72, type=int,
                        help='mini-batch size')
    parser.add_argument('-n', '--num-workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=0.001, type=float,
                        help='weight dacay')
    parser.add_argument('--lr-gamma', default=0.001, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--reg', default=0.01, type=float,
                        help='regularization term of partial entropy optimal transport')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of training epochs')
    parser.add_argument('--step', default=1000, type=int,
                        help='number of iterations per epoch')
    parser.add_argument('--interval', default=100, type=int,
                        help='print frequency')
    parser.add_argument('--method-name', default='PPOT', type=str,
                        help='method label written in the Method column of the '
                             'per-task results table (e.g. PPOT, m-PPOT)')
    parser.add_argument('--threshold', default=0.75, type=float,
                        help='confidence threshold of known samples')
    parser.add_argument('--pre-step', default=1000, type=int,
                        help='number of iterations in fine-tune step')
    parser.add_argument('--pre-lr', default=0.0005, type=float,
                        help='initial learning rate in fine-tune step, only work when use --new-opt')
    parser.add_argument('--p-entropy', default=0.01, type=float,
                        help='hyper-parameter of positive entropy loss')
    parser.add_argument('--n-entropy', default=2, type=float,
                        help='hyper-parameter of negative entropy loss')
    parser.add_argument('--ot', default=5, type=float,
                        help='hyper-parameter of ot loss')
    parser.add_argument('--neg', default=0.25, type=float,
                        help='ratio of samples in target domain to compute negative entropy loss')
    parser.add_argument('--seed', default=1024, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--checkpoint', default='',
                        help='root of network checkpoint')
    parser.add_argument('--tau1', default=0.9, type=float,
                        help='threshold of high confidence in updating alpha')
    parser.add_argument('--tau2', default=1, type=float,
                        help='threshold of known class in updating beta')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='update ratio of source prototype')
    parser.add_argument('--alpha', default=0.001, type=float,
                        help='update ratio of alpha')
    parser.add_argument('--beta', default=0.01, type=float,
                        help='update ratio of beta')
    parser.add_argument('--new-opt', action='store_true',
                        help='use new optimizer in fine-tune step')
    parser.add_argument('--balanced', action='store_true',
                        help='use balanced batch sampler in our experiment')
    parser.add_argument('--no-ssl', action='store_true',
                        help='if you do not want to pre-train network by self-supervised learning')
    parser.add_argument('--closed-set', action='store_true',
                        help='override (common, src_private, tgt_private) so all classes are common '
                             '— for closed-set DA comparison with methods like m-KPOT')

    # moco configs
    parser.add_argument('--moco-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--mlr', '--moco-learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='mlr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--mwd', '--moco-weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=200, type=int,
                        metavar='N', help='MoCo print frequency in iterations '
                                          '(~1190 iters/epoch on VisDA, so 200 -> ~6 lines/epoch)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default='True',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2304, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.2, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true', default='True',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true', default='True',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true', default='True',
                        help='use cosine lr schedule')
    parser.add_argument('--freq', default=200, type=int,
                        metavar='N', help='save frequency (default: 10)')
    parser.add_argument('--moco-batch-size', default=128, type=int)

    args = parser.parse_args()
    main(args)
