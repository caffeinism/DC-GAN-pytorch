import argparse

parser = argparse.ArgumentParser("DCGAN")
parser.add_argument('--dataset_dir', type=str, default='') # dataset directory
parser.add_argument('--result_dir', type=str, default='') # log image directory
parser.add_argument('--batch_size', type=int, default=32) # batch size
parser.add_argument('--n_epoch', type=int, default=20) # epoch size
parser.add_argument('--n_cpu', type=int, default=4) # num of process(for use worker)
parser.add_argument('--log_iter', type=int, default=1000) # print log message and save image per log_iter
parser.add_argument('--nz', type=int, default=100)  # noise dimension
parser.add_argument('--nc', type=int, default=3)    # input and out channel
parser.add_argument('--ndf', type=int, default=64)  # number of Discriminator's feature map dimension
parser.add_argument('--ngf', type=int, default=64)  # number of Generator's feature map dimension
parser.add_argument('--lr', type=float, default=0.0002) # learning rate

parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--criterion', type=str, default='BCE') # BCE / MSE
parser.add_argument('--tanh', action='store_true') # Use tanh end of generator
config, _ = parser.parse_known_args()