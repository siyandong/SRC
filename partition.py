import argparse, time
from scene_space_partition.scene_space_partition_7S import SceneSpacePartition_7S
from scene_space_partition.scene_space_partition_Cambridge import SceneSpacePartition_Cambridge
from scene_space_partition.scene_space_partition_12S import SceneSpacePartition_12S
from utils import *
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='partition the scene into hierarchical regions and prepare training data.')
    parser.add_argument('--data_path', type=str, default='./datasets', help='path to the dataset.')
    parser.add_argument('--dataset', type=str, default='7S', choices=('7S', 'Cambridge', '12S'), help='choose a dataset.')
    parser.add_argument('--scene', type=str, default='chess', help='choose a scene from the dataset.')
    parser.add_argument('--training_info', type=str, default='train_siyan_skip200.txt', help='the file that contains the list of training images.')
    parser.add_argument('--n_class', type=int, default=64, help='number of classes each level.')
    parser.add_argument('--label_validation_test', type=str2bool, default=False, help='if label the validation and test sets or not.')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='if use gpu or not.')
    args = parser.parse_args()

    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin', 'redkitchen','stairs']:
            print('selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
            print('selected scene is not valid.')
            sys.exit()

    if args.dataset == '12S':
        if args.scene not in ['apt1/kitchen', 'apt1/living', 'apt2/bed',
                              'apt2/kitchen', 'apt2/living', 'apt2/luke', 
                              'office1/gates362', 'office1/gates381', 
                              'office1/lounge', 'office1/manolis',
                              'office2/5a', 'office2/5b']:
            print('Selected scene is not valid.')
            sys.exit()


    if args.dataset == '7S':
        ssp = SceneSpacePartition_7S(
            dataset_path=args.data_path, 
            scene=args.scene, 
            train_file_path='{}/7Scenes/{}'.format(args.data_path, args.training_info), 
            n_class=args.n_class, 
            label_validation_test=args.label_validation_test)
    elif args.dataset == 'Cambridge':
        ssp = SceneSpacePartition_Cambridge(
            dataset_path=args.data_path, 
            scene=args.scene, 
            train_file_path='{}/Cambridge/{}'.format(args.data_path, args.training_info), 
            n_class=args.n_class,
            use_gpu=args.use_gpu)
    elif args.dataset == '12S':
        ssp = SceneSpacePartition_12S(
            dataset_path=args.data_path, 
            scene=args.scene, 
            train_file_path='{}/12Scenes/{}'.format(args.data_path, args.training_info), 
            n_class=args.n_class)


    # hierarchical partition.
    t_beg = time.time()
    ssp.run()
    t_end = time.time()
    print('done. total time {:.2f}s'.format(t_end-t_beg))

