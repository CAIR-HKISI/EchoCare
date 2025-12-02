import argparse
import os
from time import time
import torch
import torch.distributed as dist
import torch.optim as optim
from models.echocare_head import EchoCareHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.nn.parallel import DistributedDataParallel
from utils.data_utils import get_loader
from utils.logger import create_logger
from utils.utils import AverageMeter
from utils.atlas import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, epoch, train_loader):
        model.train()
        loss_train = []
        run_loss = AverageMeter()
        rec_avg, hiera_avg = AverageMeter(), AverageMeter()

        t1 = time()
        for step, batch in enumerate(train_loader):

            image, patch_mask, image_label = batch
            image = image.cuda(non_blocking=True)
            patch_mask = patch_mask.cuda(non_blocking=True)
            image_label = image_label.cuda(non_blocking=True)

            image_rec, hiera_pred, image_mask = model(image, patch_mask)
            loss_rec, loss_hiera = EchoCareHead.loss(image, image_label, image_mask, image_rec, hiera_pred)
            loss = loss_rec + loss_hiera

            loss_train.append(loss.item())

            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if args.lrdecay:
                scheduler.step()

            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)

            rec_avg.update(loss_rec.item(), n=args.batch_size)
            hiera_avg.update(loss_hiera.item(), n=args.batch_size)

        if dist.get_rank() == 0:
            logger.info('Loss:{}, Rec:{}, Hiera:{}'.format(run_loss.avg, rec_avg.avg, hiera_avg.avg))
            logger.info("Epoch:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(epoch, args.epochs, run_loss.avg, time() - t1))

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == (args.epochs - 1)):
            torch.save(model.state_dict(),  os.path.join(logdir, f'ckp_{epoch}.pt'))

        checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_ckp(checkpoint, os.path.join(logdir, "model_current_epoch.pt"))

        return epoch + 1

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--datadir", default="datas", type=str, help="directory to load images")
    parser.add_argument("--epochs", default=800, type=int, help="number of training epochs")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=128, type=int, help="embedding size")
    parser.add_argument("--patch_size", default=2, type=int, help="patch size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
    parser.add_argument("--batch_size", default=192, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--save_freq", default=10, type=int, help="frequency of save checkpoint")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    args = parser.parse_args()
    logdir = args.logdir

    torch.cuda.set_device(0)

    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ["RANK"])

    args.local_rank = int(os.environ["LOCAL_RANK"])
    print('arg.local_rank', args.local_rank)
    args.device = "cuda:%d" % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    print("Training in distributed mode, 1 GPU per process. Process %d, total %d." % (args.rank, args.world_size))

    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    logger = create_logger(output_dir=logdir, dist_rank=dist.get_rank(), name='echocare')

    logger.info(f"Creating model...")
    n_hiera_class = len(AtlasNode.get_all())
    model = EchoCareHead(args, n_hiera_class=n_hiera_class)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    logger.info(f"Creating {args.opt} optimizer...")
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, amsgrad=True)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    logger.info(f"Loading training dataset...")
    train_loader = get_loader(args)

    epoch = 0
    if args.resume:
        print('resume from previous checkpoints')
        model_pth = args.resume
        model_dict = torch.load(model_pth, weights_only=False)
        model.load_state_dict(model_dict["state_dict"], strict=False)
        epoch = model_dict["epoch"]
        optimizer = model_dict["optimizer"]

    if args.lrdecay:
        num_steps = len(train_loader) * args.epochs
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_steps)
        elif args.lr_schedule == "poly":
            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    while epoch < args.epochs:
        epoch = train(args, epoch, train_loader)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(logdir, "final_model.pth"))
    dist.destroy_process_group()

    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == '__main__':
    main()

