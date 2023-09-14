import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import os
from src.data.dataset import AVDataset
from src.models.Lip_reader import Lipreading
import glob
import editdistance
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="preprocessed_data_path")
    parser.add_argument('--split_file', default="./src/data/LRS2/test.ref")
    parser.add_argument('--data_type', default="LRS2")
    parser.add_argument('--model_conf', default="./src/models/model.json")

    parser.add_argument('--results_path', default='./test_results.txt')

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--rnnlm", type=str, default='./checkpoints/LM/model.pth')
    parser.add_argument("--rnnlm_conf", type=str, default='./checkpoints/LM/model.json')

    parser.add_argument("--beam_size", type=int, default=40)
    parser.add_argument("--penalty", type=float, default=0.5)
    parser.add_argument("--maxlenratio", type=float, default=0)
    parser.add_argument("--minlenratio", type=float, default=0)
    parser.add_argument("--ctc_weight", type=float, default=0.1)
    parser.add_argument("--lm_weight", type=float, default=0.5)

    parser.add_argument("--architecture", default='AVRelScore', help='AVRelScore, VCAFE, Conformer')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    return args

def main(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.model_conf, "rb") as f:
        confs = json.load(f)
    if isinstance(confs, dict):
        model_args = confs
    else:
        _, odim, model_args = confs
    model_args = argparse.Namespace(**model_args)

    model = E2E(odim, model_args, architecture=args.architecture)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    
    model.cuda()
    model.eval()

    test_data = AVDataset(
        data_path=args.data_path,
        split_file=args.split_file,
        mode='test',
        data_type=args.data_type,
    )

    Lip_reader = Lipreading(args, odim, model, test_data.char_list)
    test(Lip_reader, test_data)

def test(Lip_reader, test_data):
    wer_list = AverageMeter()
    cer_list = AverageMeter()
    with torch.no_grad():
        with open(args.split_file, 'r') as txt:
            lines = txt.readlines()
        for idx, line in enumerate(lines):
            basename, groundtruth = line.split()[0], " ".join(line.split()[1:])
            data_filename = os.path.join(args.data_path, 'Video', basename + '.mp4')
            data_aud_filename = os.path.join(args.data_path, 'Audio', basename + '.wav')

            vid, aud = test_data.load_data(data_filename, data_aud_filename)
            output = Lip_reader.predict(vid, aud)
            if isinstance(output, str):
                print(f"hyp: {output}")
                if groundtruth is not None:
                    print(f"ref: {groundtruth}")
                    wer_list.update(*get_wer(output, groundtruth))
                    cer_list.update(*get_cer(output, groundtruth))
                    print(
                        f"progress: {idx + 1}/{len(lines)}\tcur WER: {wer_list.val * 100:.1f}\t"
                        f"cur CER: {cer_list.val * 100:.1f}\t"
                        f"avg WER: {wer_list.avg * 100:.1f}\tavg CER: {cer_list.avg * 100:.1f}"
                    )

        if args.results_path is not None:
            with open(args.results_path, 'w') as txt:
                txt.write(f'WER: {wer_list.avg * 100:.3f}, CER: {cer_list.avg * 100:.3f}')

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
    main(args)

