#!/usr/bin/env python3

import argparse 

def build_interface():

    parser = argparse.ArgumentParser(
        prog="Image Classification",
        description="Defines and trains a forward feed neural network. Returns weights to disc.",
        allow_abbrev=True
    )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=300, 
        help='Default: 300'
    )

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help='Default: 32'
    )

    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.0001, 
        help='Default: 0.0001'
    )
    
    parser.add_argument(
        '--save',
        type=bool,
        default=True,
        help='Save model weights to local directory'
    )

    return parser.parse_args()

if __name__ == '__main__': pass



