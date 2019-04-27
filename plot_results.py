import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import CARETDOWN
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Incremental learning')
parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                    help='CSV file name containing results')
parser.add_argument('--type', type=str, default='single', choices=['single', 'multi', 'class', 'expanded', 'hist'], nargs='+',
                    help='Type of plot to make: single=test accuracy from single file; multi=test accuracy from several'
                         'files; class=test accuracy per class for single file')
parser.add_argument('--experiment_id', type=str, default='CRIB Toys-50',
                    help='Describe the experiment whose results are being plotted')

# Class Accuracy Args
parser.add_argument('--smoothing_margin', type=int, default=3,
                    help='Temporal margin over which to average class accuracies')
parser.add_argument('--n_classes', type=int, default=5, help='Number of classes to display')
parser.add_argument('--range', type=int, nargs=2, default=[0, 20],
                    help='Specify the margin to plot')

# Accuracy Histogram Args
parser.add_argument('--n_bins', type=int, default=50, help="Number of bins in the histogram")

savefig = True
TITLE_SIZE = 28
AXIS_SIZE = 20
matplotlib.rcParams['font.family'] = ['serif']

matplotlib.rc('axes', titlesize=TITLE_SIZE)
matplotlib.rc('axes', labelsize=AXIS_SIZE)
dotted_line_width = 2.5
anchor = (0.65, 0.43)


# TODO
def hist_class_exposure_distance(args):
    datafile = args.files[0]
    fig, ax1 = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    classes = info_classes['classes_seen']

    idxs = {}
    for i, c in enumerate(classes):
        if c in idxs:
            idxs[c] += [i]
        else:
            idxs[c] = [i]
    avg_dists = {}
    for c, idx in idxs.items():
        pass #avg_dists[c] = (idx[-1] - idx[0]) /


def hist_final_class_accs(args):
    datafile = args.files[0]
    fig, axis = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
    info_matr = np.load(matr_file)
    acc_matr = info_matr['acc_matr']
    print(np.mean(acc_matr[:, -1], axis=0))

    os.makedirs('results', exist_ok=True)
    filename = 'results/%s-histogram-class-accs.pdf' % datafile.split('.')[0].split('/')[-1]

    plt.title(args.experiment_id + ' Class Acc. after Training')

    axis.set_xlabel('% Test accuracy')
    axis.hist(acc_matr[:, -1], args.n_bins)
    axis.grid()

    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.01 ,transparent=True)


def plot_class_accs(args):
    datafile = args.files[0]
    fig, ax1 = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    info_matr = np.load(matr_file)

    classes = info_classes['classes_seen']
    model_classes = info_classes['model_classes_seen']
    classes = classes[args.range[0]:args.range[1]]
    model_classes = model_classes[args.range[0]:args.range[1]]
    # get indices for referencing accuracies in acc_matr
    class_idxs = {c: model_classes[list(classes).index(c)] for c in classes}
    num_classes = info_matr['args'][()].total_classes
    num_le = len(classes) # Number of learning exposures

    os.makedirs('results', exist_ok=True)
    filename = 'results/%s-expanded.pdf' % datafile.split('.')[0].split('/')[-1]
    linestyle = '-'
    all_lines = []

    ax1.set_xlabel('Number of learning exposures')
    ax1.set_ylabel('% Test accuracy over seen objects')
    ax1.set_yticks(np.arange(0, 105, 10))
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.05*num_classes])
    ax2.set_ylabel('Unique objects seen (UOS)')
    ax2.set_yticks(np.arange(0, num_classes+1, num_classes//10))

    plt.xticks(np.arange(0, num_le+1, num_le//10))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid()

    acc_matr = info_matr['acc_matr'][:, args.range[0]:args.range[1]]
    counter = []
    cnt = 0
    for i in range(len(classes)):
        if classes[i] in classes[:i]:
            counter.append(cnt)
        else:
            cnt += 1
            counter.append(cnt)

    # Smooth by averaging over a margin
    def smooth(matr):
        n = args.smoothing_margin
        for i in range(num_le):
            matr[:, i] = np.mean(matr[:, max(0, i - n):min(num_le - 1, i + n)], axis=1)
        return matr

    if args.smoothing_margin > 0:
        acc_matr = smooth(acc_matr)

    test_acc = np.sum(acc_matr, axis=0) / np.array(counter)
    print(test_acc)

    classes_set = list(set(classes))
    plot_classes = [classes_set[i] for i in np.random.randint(0, len(classes_set) - 1, args.n_classes)]

    all_lines.append(ax2.plot(np.arange(0, num_le), counter, color='purple',
                              linestyle=':', linewidth=dotted_line_width,
                              label='Ground-truth UOS')[0])
    all_lines.append(ax1.plot(np.arange(0, num_le), test_acc, color='blue',
                              label='Average across seen classes', linestyle='-')[0])
    # Color background space by exposure class
    exposure_colors = {}
    bars = ax1.bar(np.arange(num_le - 1), np.array([100.0] * (num_le - 1)), width=1.0, align='edge', alpha=0.5)
    for c in classes_set:
        if c in plot_classes:
            last_plot = ax1.plot(np.arange(0, num_le), acc_matr[class_idxs[c]], label='%s Accuracy' % c, linestyle='-')[0]
            all_lines.append(last_plot)
            exposure_colors[c] = last_plot.get_color()
        else:
            exposure_colors[c] = 'white'
    for i, c in enumerate(classes[1:]):
        # only display class names if there are few enough learning exposures for them to be legible
        if args.n_classes < 30:
            ax1.text(i, 0, c, {'ha': 'left', 'va': 'bottom', 'bbox': {'fc': '0.8', 'pad': 0.2}}, rotation=20)
        # get first index of class to use for indexing class_graphs
        bars[i].set_color(exposure_colors[c])

    all_labels = [l.get_label() for l in all_lines]
    plt.title(args.experiment_id + ' Class Acc. after Training')
    ax1.legend(all_lines, all_labels, bbox_to_anchor=anchor,
               loc=2, borderaxespad=0., fancybox=True, framealpha=0.7,
               fontsize=18, numpoints=1)

    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.01 ,transparent=True)

"""
def plot_class_accs(args):
    datafile = args.files[0]
    fig, ax1 = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    info_matr = np.load(matr_file)

    classes = info_classes['classes_seen']
    model_classes = info_classes['model_classes_seen']
    # get indices for referencing accuracies in acc_matr
    class_idxs = {c: model_classes[list(classes).index(c)] for c in classes}
    num_classes = info_matr['args'][()].total_classes
    num_le = len(classes) # Number of learning exposures

    os.makedirs('results', exist_ok=True)
    filename = 'results/plot-classes.pdf'
    linestyle = '-'
    all_lines = []

    ax1.set_xlabel('Number of learning exposures')
    ax1.set_ylabel('% Test accuracy over seen objects')
    ax1.set_yticks(np.arange(0, 105, 10))
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.05*num_classes])
    ax2.set_ylabel('Unique objects seen (UOS)')
    ax2.set_yticks(np.arange(0, num_classes+1, num_classes//10))

    plt.xticks(np.arange(0, num_le+1, num_le//10))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid()

    acc_matr = info_matr['acc_matr']
    counter = []
    cnt = 0
    for i in range(len(classes)):
        if classes[i] in classes[:i]:
            counter.append(cnt)
        else:
            cnt += 1
            counter.append(cnt)

    # Smooth by averaging over a margin
    def smooth(matr):
        n = args.smoothing_margin
        for i in range(num_le):
            matr[:, i] = np.mean(matr[:, max(0, i - n):min(num_le - 1, i + n)], axis=1)
        return matr

    if args.smoothing_margin > 0:
        acc_matr = smooth(acc_matr)

    test_acc = np.sum(acc_matr, axis=0) / np.array(counter)
    print(test_acc)

    classes_set = list(set(classes))
    plot_classes = [classes_set[i] for i in np.random.randint(0, len(classes_set), args.n_classes)]

    all_lines.append(ax2.plot(np.arange(0, num_le), counter, color='purple',
                              linestyle=':', linewidth=dotted_line_width,
                              label='Ground-truth UOS')[0])
    all_lines.append(ax1.plot(np.arange(0, num_le), test_acc, color='blue',
                              label='Average across seen classes', linestyle='-')[0])
    for c in plot_classes:
        all_lines.append(ax1.plot(np.arange(0, num_le), acc_matr[class_idxs[c]],
                                  label='%s Accuracy' % c, linestyle='-')[0])


    all_labels = [l.get_label() for l in all_lines]
    plt.title(title)
    ax1.legend(all_lines, all_labels, bbox_to_anchor=anchor,
               loc=2, borderaxespad=0., fancybox=True, framealpha=0.7,
               fontsize=18, numpoints=1)

    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.01 ,transparent=True)
"""

# TODO debug
def plot_accs(args):
    datafiles = args.files
    fig, ax1 = plt.subplots()
    for datafile in datafiles:
        if not os.path.exists(datafile):
            raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    info_matr = []
    for datafile in datafiles:
        matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
        info_matr += [np.load(matr_file)]

    classes = info_classes['classes_seen']
    num_classes = info_matr[0]['args'][()].total_classes
    num_le = len(classes) # Number of learning exposures
    
    os.makedirs('results', exist_ok=True)
    filename = 'results/plot-multi.pdf'
    linestyle = '-'
    all_lines = []

    ax1.set_xlabel('Number of learning exposures')
    ax1.set_ylabel('% Test accuracy over seen objects')
    ax1.set_yticks(np.arange(0, 105, 10))
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.05*num_classes])
    ax2.set_ylabel('Unique objects seen (UOS)')
    ax2.set_yticks(np.arange(0, num_classes+1, num_classes//10))

    plt.xticks(np.arange(0, num_le+1, num_le//10))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid()

    for j, datafile in enumerate(datafiles):
        acc_matr = info_matr[j]['acc_matr']
        counter = []
        cnt = 0
        for i in range(len(classes)):
            if classes[i] in classes[:i]:
                counter.append(cnt)
            else:
                cnt += 1
                counter.append(cnt)

        test_acc = np.sum(acc_matr, axis=0) / np.array(counter)
        print(test_acc)

        all_lines.append(ax2.plot(np.arange(0, num_le), counter,
                                  linestyle=':', linewidth=dotted_line_width,
                                  label='Ground-truth UOS')[0])
        all_lines.append(ax1.plot(np.arange(0, num_le), test_acc,
                                  label=datafile.split('.')[0], linestyle='-')[0])


    all_labels = [l.get_label() for l in all_lines] 
    plt.title(args.experiment_id + ' Class Acc. after Training')
    ax1.legend(all_lines, all_labels, bbox_to_anchor=anchor, 
               loc=2, borderaxespad=0., fancybox=True, framealpha=0.7, 
               fontsize=18, numpoints=1)
        
    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                pad_inches=0.01 ,transparent=True)


def plot_acc(args):
    datafile = args.files[0]
    fig, ax1 = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    info_matr = np.load(matr_file)

    classes = info_classes['classes_seen']
    num_classes = info_matr['args'][()].total_classes
    num_le = len(classes) # Number of learning exposures

    os.makedirs('results', exist_ok=True)
    filename = 'results/%s-single.pdf' % datafile.split('.')[0].split('/')[-1]
    linestyle = '-'
    all_lines = []

    ax1.set_xlabel('Number of learning exposures')
    ax1.set_ylabel('% Test accuracy over seen objects')
    ax1.set_yticks(np.arange(0, 105, 10))
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.05*num_classes])
    ax2.set_ylabel('Unique objects seen (UOS)')
    ax2.set_yticks(np.arange(0, num_classes+1, num_classes//10))

    plt.xticks(np.arange(0, num_le+1, num_le//10))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid()

    acc_matr = info_matr['acc_matr']
    counter = []
    cnt = 0
    for i in range(len(classes)):
        if classes[i] in classes[:i]:
            counter.append(cnt)
        else:
            cnt += 1
            counter.append(cnt)

    test_acc = np.sum(acc_matr, axis=0) / np.array(counter)
    print(test_acc)

    all_lines.append(ax2.plot(np.arange(0, num_le), counter, color='purple',
                              linestyle=':', linewidth=dotted_line_width,
                              label='Ground-truth UOS')[0])
    all_lines.append(ax1.plot(np.arange(0, num_le), test_acc, color='blue',
                              label=datafile.split('.')[0].split('/')[-1], linestyle='-')[0])


    all_labels = [l.get_label() for l in all_lines]
    plt.title(args.experiment_id + ' Repeated Exposure')
    ax1.legend(all_lines, all_labels, bbox_to_anchor=anchor,
               loc=2, borderaxespad=0., fancybox=True, framealpha=0.7,
               fontsize=18, numpoints=1)

    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.01 ,transparent=True)


if __name__ == '__main__':
    args = parser.parse_args()
    if 'single' in args.type:
        plot_acc(args)
    if 'multi' in args.type:
        plot_accs(args)
    if 'class' in args.type:
        plot_class_accs(args)
    if 'hist' in args.type:
        hist_final_class_accs(args)
