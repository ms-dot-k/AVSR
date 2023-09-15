import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.dataset import AVDataset
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import torch.distributed as dist
import time
import glob
import wandb
import editdistance
from datetime import datetime
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="preprocessed_data_path")
    parser.add_argument('--split_file', default="./data/LRS2/0_600.txt")
    parser.add_argument('--data_type', default="LRS2")
    parser.add_argument('--model_conf', default="./src/models/model.json")

    parser.add_argument("--max_vid_len", type=int, default=600)
    parser.add_argument("--max_txt_len", type=int, default=150)

    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints/LRS2_AVSR')
    parser.add_argument("--v_frontend_checkpoint", type=str, default='./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar')
    parser.add_argument("--a_frontend_checkpoint", type=str, default='./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--tot_iters", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--fast_validate", default=False, action='store_true')

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--visual_corruption", default=False, action='store_true')

    parser.add_argument("--mode", type=str, default='train', help='train, test, val')

    parser.add_argument("--architecture", default='AVRelScore', help='AVRelScore, VCAFE, Conformer')

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--port", type=str, default='1234')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = args.port

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    train_data = AVDataset(
        data_path=args.data_path,
        split_file=args.split_file,
        mode=args.mode,
        data_type=args.data_type,
        max_vid_len=args.max_vid_len,
        max_txt_len=args.max_txt_len,
        visual_corruption=args.visual_corruption
    )

    with open(args.model_conf, "rb") as f:
        confs = json.load(f)
    if isinstance(confs, dict):
        model_args = confs
    else:
        _, odim, model_args = confs
    model_args = argparse.Namespace(**model_args)

    model = E2E(odim, model_args, architecture=args.architecture)
    num_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    elif args.v_frontend_checkpoint is not None and args.a_frontend_checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading vid frontend checkpoint: {args.v_frontend_checkpoint}")
            print(f"Loading aud frontend checkpoint: {args.a_frontend_checkpoint}")
        update_ckpt = {}
        checkpoint_vid = torch.load(args.v_frontend_checkpoint, map_location=lambda storage, loc: storage.cuda())
        for k, v in checkpoint_vid['model_state_dict'].items():
            if not 'tcn' in k:
                if args.architecture in ['VCAFE', 'AVRelScore']:
                    update_ckpt['encoders.frontend_vid.' + k] = v
                else:
                    update_ckpt['encoder_vid.frontend_vid.' + k] = v
        checkpoint_aud = torch.load(args.a_frontend_checkpoint, map_location=lambda storage, loc: storage.cuda())
        for k, v in checkpoint_aud['model_state_dict'].items():
            if not 'tcn' in k:
                if args.architecture in ['VCAFE', 'AVRelScore']:
                    update_ckpt['encoders.frontend_aud.' + k] = v
                else:
                    update_ckpt['encoder_aud.frontend_aud.' + k] = v
        model.load_state_dict(update_ckpt, strict=False)
        del checkpoint_vid, checkpoint_aud

    model.cuda()

    params = [{'params': model.parameters()}]    
    num_train = sum(p.numel() for p in params[0]['params'])

    if args.local_rank == 0:
        print(f'Train # of params: {num_train} / {num_model}')

    
    optimizer = get_std_opt(
        model,
        model_args.adim,
        model_args.transformer_warmup_steps,
        1.,
    )

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.dataparallel:
        model = DP(model)

    # validate(model, fast_validate=False)
    train(model, train_data, args.epochs, optimizer=optimizer, args=args)

def train(model, train_data, epochs, optimizer, args):
    best_val_wer = 1.0
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])
        if args.wandb_project is not None:
            wandbrun = wandb.init(project="AVSR", name=args.wandb_project + f'_{dt_string}')
            wandbrun.config.epochs = args.epochs
            wandbrun.config.batch_size = args.batch_size
            wandbrun.config.architecture = args.architecture
            wandbrun.config.eval_step = args.eval_step
            wandbrun.config.update_frequency = args.update_frequency
            wandbrun.config.visual_corruption = args.visual_corruption
        else:
            wandbrun = None
    else:
        writer = None
        wandbrun = None

    model.train()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda x: train_data.collate_fn(x),
    )

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        wer_list = AverageMeter()
        cer_list = AverageMeter()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            prev_time = time.time()
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
                    epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['lr']))
            vid, aud, vid_len, aud_len, text = batch

            loss, loss_debug = model(vid.cuda(), aud.cuda(), vid_len.cuda(), aud_len.cuda(), text.cuda())
            
            loss = loss / args.update_frequency
            loss.backward()
            if ((i + 1) % args.update_frequency == 0) or (i + 1 == len(dataloader)):
                step += 1
                optimizer.step()
                optimizer.zero_grad()
            else:
                continue

            loss_list.append(loss.cpu().item())

            if hasattr(model, "module"):
                ys_hat = model.module.pred_pad.argmax(dim=-1)
            else:
                ys_hat = model.pred_pad.argmax(dim=-1)

            for kk, (y_true, y_hat) in enumerate(zip(text, ys_hat)):
                eos_true = torch.where(y_true == -1)[0]
                ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
                # NOTE: padding index (-1) in y_true is used to pad y_hat
                seq_hat = [train_data.char_list[int(idx)] for idx in y_hat[:ymax]]
                groundtruth = [train_data.char_list[int(idx)] for idx in y_true[:ymax]]
                output = "".join(seq_hat).replace('<space>', " ")
                groundtruth = "".join(groundtruth).replace('<space>', " ")
                wer_list.update(*get_wer(output, groundtruth))
                cer_list.update(*get_cer(output, groundtruth))
                if step % 100 == 0 and args.local_rank == 0:
                    if kk < 5:
                        print("GT: ", groundtruth.upper())
                        print("PR: ", output.upper())

            if step % 10 == 0 and args.local_rank == 0 and writer is not None:
                for k, v in loss_debug.items():
                    writer.add_scalar(f'train/{k}', v, step)
                    if wandbrun is not None:
                        wandbrun.log({f'train/{k}': v}, step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('train/loss', loss.cpu().item(), step)
                writer.add_scalar('train/wer', wer_list.val, step)
                writer.add_scalar('train/cer', cer_list.val, step)
                if wandbrun is not None:
                    wandbrun.log({'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
                    wandbrun.log({'train/wer': wer_list.val}, step)
                    wandbrun.log({'train/cer': cer_list.val}, step)
                    wandbrun.log({'train/loss': loss.cpu().item()}, step)

            if step % args.eval_step == 0:
                logs = validate(model, epoch=epoch, writer=writer, fast_validate=args.fast_validate, wandbrun=wandbrun, step=step)

                if args.local_rank == 0:
                    print('VAL_wer: ', logs[0])
                    print('Saving checkpoint: %d' % epoch)
                    if args.dataparallel or args.distributed:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({'state_dict': state_dict},
                               os.path.join(args.checkpoint_dir, 'Epoch_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[0])))

                    if logs[0] < best_val_wer:
                        best_val_wer = logs[0]
                        bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                        for prev in bests:
                            os.remove(prev)
                        torch.save({'state_dict': state_dict},
                                   os.path.join(args.checkpoint_dir, 'Best_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[0])))
            
            if args.tot_iters is not None and step == args.tot_iters:
                if args.local_rank == 0:
                    if args.dataparallel or args.distributed:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({'state_dict': state_dict},
                                os.path.join(args.checkpoint_dir, 'Last_%05d.ckpt' % (step)))
                if args.distributed:
                    dist.barrier()
                assert 1==0, "Finishing Training, arrived total iterations"

    if args.local_rank == 0:
        print('Finishing training')


def validate(model, fast_validate=False, epoch=0, writer=None, wandbrun=None, step=0):
    with torch.no_grad():
        model.eval()
        
        val_data = AVDataset(
            data_path=args.data_path,
            split_file=f'./src/data/{args.data_type}/val.txt',
            mode='val',
            data_type=args.data_type,
            max_vid_len=args.max_vid_len,
            max_txt_len=args.max_txt_len,
            visual_corruption=args.visual_corruption
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=int(args.batch_size * 1.5),
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: val_data.collate_fn(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(10 * batch_size, int(len(dataloader.dataset)))
            max_batches = 10
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        wer_list = AverageMeter()
        cer_list = AverageMeter()

        description = 'Validation on subset of the Val dataset' if fast_validate else 'Validation'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            vid, aud, vid_len, aud_len, text = batch

            if hasattr(model, "module"):
                ys_hat = model.module.valid(vid.cuda(), aud.cuda(), vid_len.cuda(), aud_len.cuda(), text.cuda())
            else:
                ys_hat = model.valid(vid.cuda(), aud.cuda(), vid_len.cuda(), aud_len.cuda(), text.cuda())

            for kk, (y_true, y_hat) in enumerate(zip(text, ys_hat)):
                eos_true = torch.where(y_true == -1)[0]
                ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
                # NOTE: padding index (-1) in y_true is used to pad y_hat
                seq_hat = [val_data.char_list[int(idx)] for idx in y_hat[:ymax]]
                groundtruth = [val_data.char_list[int(idx)] for idx in y_true[:ymax]]
                output = "".join(seq_hat).replace('<space>', " ")
                output = output[:output.find('<eos>')]
                groundtruth = "".join(groundtruth).replace('<space>', " ")
                wer_list.update(*get_wer(output, groundtruth))
                cer_list.update(*get_cer(output, groundtruth))
                if i % 10 == 0 and args.local_rank == 0:
                    if kk < 5:
                        print("GT: ", groundtruth.upper())
                        print("PR: ", output.upper())

            if args.distributed:
                dist.barrier()

            if i >= max_batches:
                break

        if args.local_rank == 0 and writer is not None:
            writer.add_scalar('val/wer', wer_list.avg, step)
            writer.add_scalar('val/cer', cer_list.avg, step)
            if wandbrun is not None:
                wandbrun.log({'val/wer': wer_list.avg}, step)
                wandbrun.log({'val/cer': cer_list.avg}, step)

        model.train()
        return wer_list.avg, cer_list.avg

def get_wer(predict, truth):
    predict = predict.split(' ')
    truth = truth.split(' ')
    err = editdistance.eval(predict, truth)
    num = len(truth)
    return err, num

def get_cer(predict, truth):
    predict = predict.replace(' ', '')
    truth = truth.replace(' ', '')
    err = editdistance.eval(predict, truth)
    num = len(truth)
    return err, num

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, err, num=1):
        self.val = err / num
        self.sum += err
        self.count += num
        self.avg = self.sum / self.count

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

