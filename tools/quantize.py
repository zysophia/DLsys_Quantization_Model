import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='DLsys qunatize tool')
    parser.add_argument('config', help='quantize config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # TODO


if __name__ == "__main__":
    main()