import os
import argparse

from utils import load_config, build_model, save_model

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
    args = parse_args()
    # configurations
    config = load_config(args.config)
    # dataloader for inference
    dataloader = build_dataloader(config.data)
    # build model
    model = build_model(config.model)

    # quantize model
    quant_model = quantize_model(model, args.fuse-conv-bn)

    save_model(quant_model, args.out)


if __name__ == "__main__":
    main()