import argparse
import os
f_path = os.path.abspath('..')
root_path = f_path.split('3d_code')[0]

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0,1], help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=3,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
parser.add_argument('--dataset_path',default = root_path+'/datasets/LiTS17/fixed/',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = root_path+'/datasets/LiTS17/test',help='Testset path')
parser.add_argument('--save',default='TCN',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=2,help='batch size of trainset')
parser.add_argument('--dataset', type=str, default='71_heart',help='name of dataset')
# train
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--val_crop_max_size', type=int, default=96)
parser.add_argument('--pre-train', action='store_true', default=False)
parser.add_argument('--pre-train-path',type=str , default='')
parser.add_argument('--pre_epoch',type=int , default=240)
# test
parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')



parser.add_argument("--extractor", type=str, default='r2p1d',
                        choices=['r2p1d', 'r2p1d_4layer', 'r3d', 'r3d_4layer', 'r2d', 'r2d_50', 'r2d_101'],
                        help='The type of CNN feature extractor. We defaulty use R(2+1)D.')
parser.add_argument("--context", type=str, default='bilstm', 
                    choices=['lstm', 'bilstm', 'gcn', 'transformer', 'none'],
                    help='The type temporal context modeling network. Default is bidirectional LSTMs.')
parser.add_argument("--aggregate", type=str, default='avgpool',
                    choices=['mean', 'avgpool', 'final', 'lstm'],
                    help='Spatiotemporal aggregation mode. Default is avgpool.')

parser.add_argument("--task", type=str, default='Across', choices=['Calot', 'Dissection', 'Across'],
                    help='We experimented on the acrossed task.')
parser.add_argument("--split_index", type=int, default=1, choices=[0,1,2,3,4])

parser.add_argument("--num_samples", type=int, default=64, help='Equals to T in the paper.')
parser.add_argument("--num_epochs", type=int, default=40)

parser.add_argument("--multi_gpu", action='store_true')

parser.add_argument("--randseed", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--schedule_step", type=int, default=20)
parser.add_argument("--freeze_extractor", action='store_true', help='The CNN extractor is freezed avoiding over-fitting.')
parser.add_argument("--scene_node", action='store_true')
parser.add_argument("--num_parts", type=int, default=1)
parser.add_argument("--no_pastpro", action='store_true')
parser.add_argument("--shaping_weight", type=float, default=10, help='Default is 10.')
parser.add_argument("--heatmap_regu_weight", type=float, default=0, help='Positional regu is not supported in the dataset.')

# Unfrequently used arguments
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--debug", action='store_true')     # not in use
parser.add_argument("--attention", action='store_true') # not in use
parser.add_argument("--avgpool_parts", action='store_true') # not in use
parser.add_argument("--multi_lstms", action='store_true')   
parser.add_argument("--prepro", action='store_true')        # not in use
parser.add_argument("--simple_pastpro", action='store_true')# not in use
parser.add_argument("--rolling_train", action='store_true') # not in use
parser.add_argument("--freeze_half_extractor", action='store_true') # not in use
parser.add_argument("--freeze_central", action='store_true')        # not in use
parser.add_argument("--init_extractor", action='store_true')        # not in use
parser.add_argument("--tconsist_start_from", type=int, default=0)   # not in use
parser.add_argument("--train_sample_augment", type=int, default=1)  # not in use
parser.add_argument("--test_sample_augment", type=int, default=1)   # not in use
parser.add_argument("--balanced_train_sample", action='store_true') # not in use
parser.add_argument("--noised_train_label", action='store_true')    # not in use

parser.add_argument("--visualize", action='store_true',
                    help='If true, the assignment maps will be saved in ./group_vis_res file')
parser.add_argument("--save_separately", action='store_true')
parser.add_argument("--save_checkpoint", action='store_true')
parser.add_argument("--extra_label", type=str, default=None)

args = parser.parse_args()