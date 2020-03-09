from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Homewrok 4')

    #setup
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')


    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../Splits/',
                        help="root path to data directory")
    parser.add_argument('--val_split', type=str, default='split3',
                        help="root path to data directory")

    #Training parameter
    parser.add_argument('--epoch', default=100, type=int,
                        help="num of epochs iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="every val_epoch epochs runs validation")
    parser.add_argument('--train_batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help="initial learning rate")


    #Output
    parser.add_argument('--save_dir', type=str, default='log')




    args = parser.parse_args()

    return args
