from model import iCaRLNet
import torch
from torch.autograd import Variable
from data_loader import iToys
import argparse
import time
import numpy as np
import cv2
import copy
import subprocess
import os
import torch.multiprocessing as mp
import atexit
import sys
from data_generator.data_generator import DataGenerator
from data_generator.all_classes import classes

parser = argparse.ArgumentParser(description='Continuum learning')

# Saving options
parser.add_argument('--outfile', default='toys_results/temp.csv', type=str, 
                    help='Output file name')
parser.add_argument('--save_all', dest='save_all', action='store_true', 
                    help='Option to save models after each '
                         'test_freq number of learning exposures')
parser.add_argument('--save_all_dir', dest='save_all_dir', type=str, 
                    default=None, help='Directory to store all models in')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='Resume training from checkpoint at outfile')
parser.add_argument('--resume_outfile', default='no_resume', type=str, 
                    help='Output file name after resuming')

# Hyperparameters
parser.add_argument('--init_lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--init_lr_ft', default=0.001, type=float, 
                    help='Init learning rate for fine tuning (for E2E)')
parser.add_argument('--num_epoch', default=15, type=int, 
                    help='Number of epochs')
parser.add_argument('--num_epoch_ft', default=10, type=int, 
                    help='Number of epochs for finetuning (for E2E)')
parser.add_argument('--lrd', default=10.0, type=float, 
                    help='Learning rate decrease factor')
parser.add_argument('--wd', default=0.0001, type=float, 
                    help='Weight decay for SGD')
parser.add_argument('--batch_size', default=200, type=int, 
                    help='Mini batch size for training')
parser.add_argument('--llr_freq', default=10, type=int, 
                    help='Learning rate lowering frequency for SGD (for E2E)')
parser.add_argument('--batch_size_test', default=200, type=int, 
                    help='Mini batch size for testing')

# CRIB options
parser.add_argument('--dunit_length', default=100, type=int, 
                    help='Dataunit length')
parser.add_argument('--num_exemplars', default=600, type=int,
                    help='number of exemplars')
parser.add_argument('--img_size', default=224, type=int, 
                    help='Size of images input to the network')
parser.add_argument('--rendered_img_size', default=300, type=int, 
                    help='Size of rendered images')
parser.add_argument('--total_classes', default=200, type=int, 
                    help='Total number of classes')
parser.add_argument('--num_iters', default=200, type=int, 
                    help='Number of iterations')

# Model options
parser.add_argument('--algo', default='icarl', type=str, 
                    help='Algorithm to run. Options : icarl, e2e, lwf')
parser.add_argument('--no_dist', dest='dist', action='store_false',
                    help='Option to switch off distillation loss')
parser.add_argument('--pt', dest='pretrained', action='store_true',
                    help='Option to start from an ImageNet pretrained model')
parser.add_argument('--ncm', dest='ncm', action='store_true',
                    help='Use nearest class mean classification (for E2E)')

# Training options
parser.add_argument('--diff_order', dest='d_order', action='store_true',
                    help='Use a random order of classes introduced')
parser.add_argument('--pre_augment', dest='pre_augment', action='store_true', 
                    help='Whether to augment the dataset just once'
                         ' before training starts (for E2E)')
parser.add_argument('--no_jitter', dest='jitter', action='store_false',
                    help='Option for no color jittering (for iCaRL)')
parser.add_argument('--h_ch', default=0.02, type=float, 
                    help='Color jittering : max hue change')
parser.add_argument('--s_ch', default=0.05, type=float, 
                    help='Color jittering : max saturation change')
parser.add_argument('--l_ch', default=0.1, type=float, 
                    help='Color jittering : max lightness change')

# System options
parser.add_argument('--test_freq', default=1, type=int, 
                    help='Number of iterations of training after'
                         ' which a test is done/model saved')
parser.add_argument('--num_workers', default=8, type=int, 
                    help='Maximum number of threads spawned')
parser.add_argument('--one_gpu', dest='one_gpu', action='store_true', 
                    help='Option to run multiprocessing on 1 GPU')


parser.set_defaults(pre_augment=False)
parser.set_defaults(ncm=False)
parser.set_defaults(dist=True)
parser.set_defaults(pretrained=False)
parser.set_defaults(d_order=False)
parser.set_defaults(jitter=True)
parser.set_defaults(save_all=False)
parser.set_defaults(resume=False)
parser.set_defaults(one_gpu=False)

def main():
    args = parser.parse_args()
    torch.backends.cudnn.benchmark=True
    mp.set_sharing_strategy('file_system')

    # multiprocessing on a single GPU
    if args.one_gpu:
        mp.get_context('spawn')
    
    # Hyper Parameters
    total_classes = args.total_classes
    batch_size = args.batch_size
    num_iters = args.num_iters
    rendered_img_size = args.rendered_img_size
    dunit_length      = args.dunit_length
    points_on_vsphere = 3
    img_size          = args.img_size
    # size of random viewing sphere samples used for measuring sequential learning accuracy (not a walk on the viewing sphere, uniformly random views)
    SIZE_TEST = 100
    #loading mean image; resizing to rendered image size if necessary
    mean_image = np.load('data_generator/mean_image.npy')
    mean_image.astype(np.float32)
    mean_image = cv2.resize(mean_image, (rendered_img_size, rendered_img_size))

    # To pass to dataloaders for preallocation
    max_train_data_size = 2 * dunit_length + args.num_exemplars
    max_test_data_size = total_classes * SIZE_TEST

    if args.save_all:
        if args.save_all_dir is None:
            raise Exception("Directory to save all model not provided")
        os.makedirs('%s-saved_models'%(os.path.join(args.save_all_dir, (args.outfile).split('.')[0])), exist_ok=True)


    # permutation of classes
    if args.d_order == True:
        perm_id = np.random.permutation(args.total_classes)
    else:
        perm_id = np.arange(args.total_classes)

    # NOTE : a permutation file should be present for repeated exposures expt
    # this is to ensure that over multiple runs permutations are generated the 
    # way described in Experiments section of the paper
    # If there are repeated exposures
    if args.num_iters > args.total_classes:
        perm_id_all = np.load('permutation_files/permutation_%d_%d.npy'%(args.total_classes, args.num_iters//args.total_classes))
        for i in range(len(perm_id_all)):
            perm_id_all[i] = perm_id[perm_id_all[i]]
        perm_id = perm_id_all


    # GPU indices
    train_device = 0
    test_device = 1

    if args.one_gpu:
        test_device = 0

    # Initialize CNN
    K = args.num_exemplars # total number of exemplars
    icarl = iCaRLNet(1, args, device=train_device)


    expt_githash = subprocess.check_output(['git', 'describe', '--always'])

    class_map = {cl_name:idx for idx,cl_name in enumerate(classes)}
    classes = np.array(classes)
    class_occurrences = np.zeros(classes.shape)
    classes_seen = []
    icarl_classes_seen = []
    exemplar_sets = []
    all_loss_vals = []
    all_loss_val_epoc_ints = []
    all_seen = False
    # Acc matr row index represents class number and there is a row for each class even if classes are seen in batches of more than 1 at a time
    acc_matr = np.zeros((total_classes, num_iters))

    # Using this map to declare all test sets before training starts
    icarl_all_classes_map = {}
    unique_counter = 0
    for i, cl in enumerate(classes[perm_id]):
        if cl not in icarl_all_classes_map:
            icarl_all_classes_map[cl] = unique_counter
            unique_counter += 1
    icarl_all_classes = {i:cl for cl,i in icarl_all_classes_map.items()}


    # Using the following data generators we can keep track of coverages for each class
    data_generators = [DataGenerator(model_name = classes[i], n_frames = dunit_length, size_test = SIZE_TEST, resolution = rendered_img_size) for i in range(total_classes)]

    # Declare empty training and test sets
    test_set = iToys(args, img_size, mean_image, data_generators = [], max_data_size = max_test_data_size)
    train_set = None

    # Conditional variable, shared vars for synchronization
    cond_var = mp.Condition()
    train_counter = mp.Value('i', 0)
    test_counter = mp.Value('i', 0)
    dataQueue = mp.Queue()
    all_done = mp.Event()
    data_mgr = mp.Manager()
    expanded_classes = data_mgr.list([None for i in range(args.test_freq)])

    if args.resume:
        print('resuming model from %s-model.pth.tar'%(args.outfile).split('.')[0])
        
        icarl = torch.load('%s-model.pth.tar'%(args.outfile).split('.')[0], map_location=lambda storage, loc: storage)
        icarl.device = train_device

        icarl.exemplar_means = []
        icarl.compute_means = True

        info_classes = np.load('%s-classes.npz'%(args.outfile).split('.')[0])
        info_matr = np.load('%s-matr.npz'%(args.outfile).split('.')[0])
        if expt_githash != info_classes['expt_githash']:
            print('Warning : Code was changed since last time model was saved')
            print('Last commit hash : ', info_classes['expt_githash'])
            print('Current commit hash : ', expt_githash)

        if args.resume_outfile != 'no_resume':
            args.outfile = args.resume_outfile
        else:
            print('######## Warning : overwriting old files ########')

        num_iters_done = info_matr['num_iters_done']

        acc_matr = info_matr['acc_matr']
        resume_params = info_matr['other_hyper_params'].item()
        perm_id = resume_params['perm_id']

        icarl_classes_seen = list(info_classes['icarl_classes_seen'][:num_iters_done])
        classes_seen = list(info_classes['classes_seen'][:num_iters_done])
        all_loss_vals = list(info_classes['loss_vals'][:num_iters_done])
        all_loss_val_epoc_ints = list(info_classes['loss_val_epoch_ints'][:num_iters_done])

        # for continuing
        if args.cont:
            extra_iters = args.num_iters-len(perm_id)

            perm_id_new = np.zeros(args.num_iters, dtype=np.int32)
            perm_id_new[:len(perm_id)] = perm_id
            perm_id_all_cont = np.load('permutation_files/permutation_%d_%d.npy'%(args.total_classes, extra_iters//args.total_classes))
            if args.d_order == True:
                perm_id_extra = np.random.permutation(args.total_classes)
            else:
                perm_id_extra = np.arange(args.total_classes)
            for i in range(len(perm_id_all_cont)):
                perm_id_all_cont[i] = perm_id_extra[perm_id_all_cont[i]]
            perm_id_new[len(perm_id):] = perm_id_all_cont
            perm_id = perm_id_new
            acc_matr_new = np.zeros((args.total_classes, args.num_iters))
            acc_matr_new[:,:len(classes_seen)] = acc_matr
            acc_matr = acc_matr_new


        args.total_classes = total_classes = resume_params['total_classes']
        args.batch_size = batch_size = resume_params['batch_size']
        if not args.cont:
            args.num_iters = num_iters = resume_params['num_iters']

        args.dunit_length = dunit_length = resume_params['dunit_length']
        args.num_exemplars = K = resume_params['num_exemplars']
        args.img_size = img_size = resume_params['img_size']
        args.dist = resume_params['dist']
        args.algo = resume_params['algo']
        args.finetuned = resume_params['finetuned']
        args.rendered_img_size = resume_params['rendered_img_size']
        args.img_size = resume_params['img_size']
        args.pretrained = resume_params['pretrained']
        args.rgb = resume_params['rgb']
        args.d_order = resume_params['d_order']
        args.h_ch = resume_params['h_ch']
        args.s_ch = resume_params['s_ch']
        args.l_ch = resume_params['l_ch']

        points_on_vsphere = resume_params['points_on_vsphere']
        if not args.cont:
            perm_id = resume_params['perm_id']


        train_counter = mp.Value('i', len(classes_seen))
        test_counter = mp.Value('i', len(classes_seen))

        # expanding test set to everything seen earlier
        for cl in icarl.classes:
            print('Expanding class for resuming : ', cl)
            test_set.expand(args, img_size, [data_generators[class_map[cl]]], \
                            [cl], icarl.classes_map, 'test', SIZE_TEST)

        # Get the datagenerators state upto resuming point
        for cl in classes_seen:
            data_generators[class_map[cl]].n_exposures += 1

        # Needed after model reload, for some reason
        for p in icarl.parameters():
            p.requires_grad = True

    other_hyper_params = {'total_classes' : total_classes,
                          'batch_size' : batch_size,
                          'num_iters' : num_iters,
                          'dunit_length' : dunit_length,
                          'num_exemplars' : K,
                          'img_size' : img_size,
                          'points_on_vsphere' : points_on_vsphere,
                          'dist' : args.dist,
                          'algo' : args.algo,
                          'finetuned' : args.finetuned,
                          'rendered_img_size' : args.rendered_img_size,
                          'img_size' : args.img_size,
                          'pretrained' : args.pretrained,
                          'rgb' : args.rgb,
                          'd_order' : args.d_order,
                          'perm_id' : perm_id,
                          'h_ch' : args.h_ch,
                          's_ch' : args.s_ch,
                          'l_ch' : args.l_ch}

    train_process = mp.Process(target=train_run, args=(train_device,))
    test_process = mp.Process(target=test_run, args=(test_device,))
    atexit.register(cleanup, train_process, test_process)
    train_process.start()
    test_process.start()

    train_process.join()
    print('Train Process Completed')
    test_process.join()
    print('Test Process Completed')



def train_run(device):
    # Timers
    global train_set
    global train_counter
    global test_counter
    global expanded_classes
    icarl.cuda(device = device)
    s = len(classes_seen)
    print('####### Train Process Running ########')
    print('Info run: ', other_hyper_params)
    print('Info run: ', icarl.fetch_hyper_params())
    train_wait_time = 0

    while s < args.num_iters:
        
        time_ptr = time.time()
        # Do not start training till testing catches up
        cond_var.acquire()
        # while to avoid spurious wakeups
        while test_counter.value + args.test_freq <= train_counter.value:
            print('[Train Process] Waiting on test process')
            print('[Train Process] train_counter : ', train_counter.value)
            print('[Train Process] test_counter : ', test_counter.value)
            print('[Train Process] icarl n_classes : ', icarl.n_classes)
            cond_var.wait()
        cond_var.release()
        train_wait_time += time.time() - time_ptr

        curr_class_idx = perm_id[s]
        curr_class = classes[curr_class_idx]
        class_occurrences[curr_class_idx] += 1
        classes_seen.append(curr_class)
        # Boolean to store if the current iteration saw a new class
        curr_expanded = False
        
        if curr_class not in icarl.classes_map:
            icarl.increment_classes([curr_class])
            icarl.cuda(device=device)
            curr_expanded = True
            

        icarl_curr_class_idx = icarl.classes_map[curr_class]
        icarl_classes_seen.append(icarl_curr_class_idx)

        # Load Datasets
        print("Loading training examples for class index %d , %s, at iteration %d"%(icarl_curr_class_idx, curr_class, s))
        
        if train_set is None:
            train_set = iToys(args, img_size, mean_image, [data_generators[curr_class_idx]], max_train_data_size, [curr_class], icarl.classes_map, 'train', SIZE_TEST, du_idx=s)
        else:
            train_set.pseudo_init(args, img_size, [data_generators[curr_class_idx]], [curr_class], icarl.classes_map, 'train', SIZE_TEST, du_idx=s)
        
        # Keep a copy of previous model for distillation
        prev_model = copy.deepcopy(icarl)
        prev_model.cuda(device=device)
        for p in prev_model.parameters():
            p.requires_grad = False

        icarl.train()
        
        # Update representation via BackProp
        loss_vals, loss_val_epoch_ints = icarl.update_representation(train_set, prev_model, [icarl_curr_class_idx], args)
        all_loss_vals.append(list(loss_vals))
        all_loss_val_epoc_ints.append(list(loss_val_epoch_ints))
        del prev_model

        m = int(K / icarl.n_classes)
        icarl.eval()
        
        if args.algo == 'icarl' or args.algo == 'hybrid1':
            # Reduce exemplar sets for known classes
            icarl.reduce_exemplar_sets(m)

            # Construct exemplar sets for current class
            print ("Constructing exemplar set for class index %d , %s ..."%(icarl_curr_class_idx, curr_class), end="")
            images, du_maps, image_bbs = train_set.get_image_class(icarl_curr_class_idx)
            icarl.construct_exemplar_set(images, du_maps, image_bbs, m, icarl_curr_class_idx, s)
            print ("Done")

            # list() is needed to append a copy of list
            exemplar_sets.append(list(icarl.eset_du_maps))

            for y, P_y in enumerate(icarl.exemplar_sets):
                print ("Exemplar set for class-%d:" % (y), P_y.shape)

        icarl.n_known = icarl.n_classes
        print ("Icarl num classes : %d, " % icarl.n_known)
        print ("Icarl classes : ", icarl.classes)

        cond_var.acquire()
        train_counter.value += 1
        if curr_expanded:
            expanded_classes[s%args.test_freq] = (curr_class_idx, curr_class)
        else:
            expanded_classes[s%args.test_freq] = None
            
        print('[Train Process] expanded_classes : ', expanded_classes)
        if train_counter.value == test_counter.value + args.test_freq:
            temp_model = copy.deepcopy(icarl)
            temp_model.cpu()
            dataQueue.put(temp_model)
        cond_var.notify_all()
        cond_var.release()

        np.savez('%s-classes.npz'%(args.outfile)[:-4], icarl_classes_seen = icarl_classes_seen, classes_seen = classes_seen, expt_githash = expt_githash, exemplar_sets = np.array(exemplar_sets), loss_vals = np.array(all_loss_vals), loss_val_epoch_ints = np.array(all_loss_val_epoc_ints))

        # loop var increment
        s += 1 

    time_ptr = time.time()
    all_done.wait()
    train_wait_time += time.time() - time_ptr
    print('[Train Process] Done, total time spent waiting : ', train_wait_time)



def test_run(device):
    global test_set
    global train_counter
    global test_counter
    global expanded_classes
    print('####### Test Process Running ########')
    test_icarl = None
    s = args.test_freq * (len(classes_seen)//args.test_freq)

    test_wait_time = 0
    with open(args.outfile, 'w') as file:
        print("Icarl classes, Train Accuracy, Test Accuracy", file=file)
        while s < args.num_iters:

            # Wait till training is done
            time_ptr = time.time()
            cond_var.acquire()
            while train_counter.value < test_counter.value + args.test_freq:
                print('[Test Process] Waiting on train process')
                print('[Test Process] train_counter : ', train_counter.value)
                print('[Test Process] test_counter : ', test_counter.value)
                if test_icarl is not None:
                    print('[Test Process] test_icarl n_classes : ', test_icarl.n_classes)
                cond_var.wait()
            cond_var.release()
            test_wait_time += time.time() - time_ptr

            cond_var.acquire()
            test_icarl = dataQueue.get()
            expanded_classes_copy = copy.deepcopy(expanded_classes)
            test_counter.value += args.test_freq
            cond_var.notify_all()
            cond_var.release()

            # NOTE : test set only needs to be expanded when a new exposure is seen
            for expanded_class in expanded_classes_copy:
                if expanded_class is not None:
                    idx, cl = expanded_class
                    print('[Test Process] Loading test data')
                    test_set.expand(args, img_size, [data_generators[idx]], \
                                        [cl], test_icarl.classes_map, 'test', SIZE_TEST)
            
            print("[Test Process] Test Set Length:", test_set.curr_len)

            test_icarl.device = device
            test_icarl.cuda(device=device)
            test_icarl.eval()
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            print ("%d, " % test_icarl.n_known, end="", file=file)

            print ("[Test Process] Computing Accuracy matrix...")
            

            # Accuracy matrix

            ############# Test Accuracy computation ###########
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for indices, images, labels in test_loader:
                    images = Variable(images).cuda(device=device)
                    preds = test_icarl.classify(images)
                    all_preds.append(preds.data.cpu().numpy())
                    all_labels.append(labels.numpy())
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            for i in range(test_icarl.n_known):
                class_preds = all_preds[all_labels == i]
                correct = np.sum(class_preds == i)
                total = len(class_preds)
                acc_matr[i, s] = (100.0 * correct/total)

            test_acc = np.mean(acc_matr[:test_icarl.n_known,s])
            print ('%.2f ,' % test_acc, file=file)
            print ("[Test Process] Accuracy matrix: \n", acc_matr[:min(test_icarl.n_known, 10), :min(s + args.test_freq, 10)])
            print ("[Test Process] Test Accuracy after %d iterations : "%(s + args.test_freq), test_acc)

            print ("[Test Process] Saving model and other data")
            test_icarl.cpu()
            if not args.save_all:
                torch.save(test_icarl, '%s-model.pth.tar'%(args.outfile).split('.')[0])
            else:
                torch.save(test_icarl, '%s-saved_models/model_iter_%d.pth.tar'%(os.path.join(args.save_all_dir, (args.outfile).split('.')[0]), s))

            # loop var increment
            s += args.test_freq

            np.savez('%s-matr.npz'%(args.outfile).split('.')[0], acc_matr = acc_matr, icarl_hyper_params = icarl.fetch_hyper_params(), other_hyper_params = other_hyper_params, num_iters_done = s)
        
        print ("[Test Process] Done, total time spent waiting : ", test_wait_time)
        all_done.set()


def cleanup(train_process, test_process):
    train_process.terminate()
    test_process.terminate()

if __name__ == "__main__":
    main()
    

