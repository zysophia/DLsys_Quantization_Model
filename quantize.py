import os
import sys
sys.path.append('python')
import argparse

import needle as ndl
from apps.models import ResNet9
from utils import save_model
from tools import quant_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='DLsys qunatize tool')
    parser.add_argument('config', help='quantize config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        type=bool,
        default=False,
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    # args = parse_args()
    # configurations
    # config = load_config(args.config)
    # dataloader for inference
    # dataloader = build_dataloader(config.data)
    # build model
    # model = build_model(config.model)
    model = ResNet9(ndl.cpu())
    save_model(model, "checkpoints/resnet9.ndl")

    # quantize model
    q_model = quant_model(model)
    save_model(q_model, "checkpoints/resnet9-quant.ndl")

    print("Original size: ", os.path.getsize("checkpoints/resnet9.ndl")/(1024*1024))
    print("Quantize size: ", os.path.getsize("checkpoints/resnet9-quant.ndl")/(1024*1024))


if __name__ == "__main__":
    main()