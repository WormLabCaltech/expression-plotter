import numpy as np
import pandas as pd
import numba
import queue
import threading
import itertools as it
import analyzer as ana

def calculate_pvalues(df, blabel, tlabel, clabel, n, f=np.mean **kwargs):
    """
    Calculates the p value of the sample.

    Parmas:
    df     --- (pandas.DataFrame) data read from csv
    blabel --- (str) label of column to group by
    tlabel --- (str) label of column of total samples
    clabel --- (str) label of column of counts
    n      --- (int) # of bootstraps
    f      --- (function) statistic to apply

    kwargs:
    s     --- (boolean) whether to save matrix to csv (default: False)
    fname --- (str) csv file name
    ctrl  --- (str) control
    """
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)
    ctrl = kwargs.pop('ctrl', None)

    matrix = df.set_index(blabel) # set index

    # get genotypes
    genotypes = matrix.index.unique()

    p_vals = ana.make_empty_dataframe(len(genotypes),\
            len(genotypes), genotypes, genotypes) # empty pandas dataframe

    threads = []
    qu = queue.Queue()

    # for loop to iterate through all pairwise comparisons (not permutation)
    print('#Starting threads for bootstrapping...')
    # if no control is given, perform all pairwise comparisons
    if ctrl is None:
        for pair in it.combinations(genotypes, 2):

            thread = threading.Thread(target=calculate_deltas_queue,\
                                    args=(matrix, tlabel, clabel, pair[0], pair[1], n, qu))
            threads.append(thread)

            thread.setDaemon(True)
            thread.start()

    # control given
    else:
        for genotype in genotypes:
            if genotype == ctrl:
                continue

            thread = threading.Thread(target=calculate_deltas_queue,
                                    args=(matrix, tlabel, clabel, ctrl, genotype, n, qu))
            threads.append(thread)

            thread.setDaemon(True)
            thread.start()

    for thread in threads:
        gene_1, gene_2, delta_obs, deltas_bootstrapped = qu.get()
        p_vals[gene_1][gene_2] = ana.calculate_pvalue(delta_obs, deltas_bootstrapped)

    print('#Bootstrapping of {} threads complete.\n'.format(len(threads)))

    print('#P-value matrix:')
    print(p_vals)

    # save matrix to csv
    if s:
        print('#Saving p-value matrix\n')
        ana.save_matrix(p_vals, fname)

    return p_vals.astype(float)

def calculate_deltas_queue(matrix, tlabel, clabel, gene_1, gene_2, n, queue):
    """
    Function to calculate deltas with multithreading.
    Saves p values as tuples in queue.

    Params:
    matrix         --- (pandas.DataFrame) with index correctly set
    gene_1, gene_2 --- (String) genotypes to be compared
    n              --- (int) # of bootstraps
    queue          --- (queue.Queue) queue to save results
    f              --- (function) to calculate deltas

    Returns: none
    """
    matrix_1 = matrix[matrix.index == gene_1]
    matrix_2 = matrix[matrix.index == gene_2]
    ts_1 = np.array(matrix_1[tlabel])
    cs_1 = np.array(matrix_1[clabel])
    ts_2 = np.array(matrix_2[tlabel])
    cs_2 = np.array(matrix_2[clabel])

    delta_obs, deltas_bootstrapped = calculate_deltas(ts_1, cs_1, ts_2, cs_2, n)

    queue.put((gene_1, gene_2, delta_obs, deltas_bootstrapped))

def calculate_deltas(ts_1, cs_1, ts_2, cs_2, n, f=np.mean):
    """
    Calculates the observed and bootstrapped deltas.

    Params:
    ts_1 --- (np.array) total samples 1
    cs_1 --- (np.array) counts 1
    ts_2 --- (np.array) total samples 2
    cs_2 --- (np.array) counts 2
    n    --- (int) # of bootstraps
    f    --- (function) statistic to apply

    Returns:
    """
    # calculate observed delta
    mean_1 = f(cs_1 / ts_1)
    mean_2 = f(cs_2 / ts_2)
    delta_obs = mean_2 - mean_1

    deltas_bootstrapped = bootstrap_deltas(ts_1, cs_1, ts_2, cs_2, n, f)

    return delta_obs, deltas_bootstrapped

def bootstrap_deltas(ts_1, cs_1, ts_2, cs_2, n, f=np.mean):
        """
        Calculates bootstrapped deltas.

        Params:
        ts_1 --- (np.array) total samples 1
        cs_1 --- (np.array) counts 1
        ts_2 --- (np.array) total samples 2
        cs_2 --- (np.array) counts 2
        n    --- (int) # of bootstraps

        Returns:
        deltas --- (np.array) of length n
        """
        # total number of samples
        ts_n = np.sum(ts_1) + np.sum(ts_2)
        cs_n = np.sum(cs_1) + np.sum(cs_2)

        # mixed array
        mixed = np.zeros(ts_n)
        mixed[0:cs_n] = np.ones(cs_n)

        # function to be numbaized
        @numba.jit(nopython=True, nogil=True)
        def difference(ts_1, cs_1, ts_2, cs_2, n):
            """
            Calculates delta based on function f.
            """

            # initialize deltas array
            deltas = np.zeros(n)

            # perform bootstraps
            # TODO: use np.random.binomial - can it be done without looping n times?
            for i in np.arange(n):
                nullp_1 = np.zeros(len(ts_1))
                nullp_2 = np.zeros(len(ts_2))

                for j in np.arange(len(ts_1)):
                    nullc = np.sum(np.random.choice(mixed, cs_1[j], replace=True))
                    nullp_1[j] = nullc / ts_1[j]

                for j in np.arange(len(ts_2)):
                    nullc = np.sum(np.random.choice(mixed, cs_2[j], replace=True))
                    nullp_2[j] = nullc / ts_2[j]

                # calculate difference of means
                delta = f(nullp_2) - f(nullp_1)

                deltas[i] = delta

            return deltas

        deltas = difference(ts_1, cs_1, ts_2, cs_2, n)

        return deltas

if __name__ == '__main__':
    import argparse
    import os

    n = 10**4
    stat = 'mean'
    fs = {'mean': np.mean,
            'median': np.median}

    parser = argparse.ArgumentParser(description='Perform statistical analysis on binary data.')
    # begin command line arguments
    parser.add_argument('csv_data',
                        help='The full path to the csv data file.',
                        type=str)
    parser.add_argument('title',
                        help='Title for your analysis. (without file \
                        extension)',
                        type=str)
    parser.add_argument('-b',
                        help='Number of bootstraps to perform. \
                        (default: {0})'.format(n),
                        type=int,
                        default=100)
    parser.add_argument('-i',
                        help='Column to group measurements by. \
                        (defaults to first column of csv)',
                        type=str,
                        default=None)
    parser.add_argument('-t',
                        help='Column for total sample size. \
                        (defaults to second column of csv)',
                        type=str,
                        default=None)
    parser.add_argument('-c',
                        help='Column for counts. \
                        (defaults to third column of csv)',
                        type=str,
                        default=None)
    parser.add_argument('-s',
                        help='Statistic to apply. \
                        (default: {})'.format(stat),
                        type=str,
                        choices=fs.keys(),
                        default='mean')
    parser.add_argument('--save',
                        help='Save data to csv.',
                        action='store_true')
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    title = args.title
    n = args.b
    blabel = args.i
    tlabel = args.t
    clabel = args.c
    f = fs[args.s]
    s = args.save

    df = pd.read_csv(csv_path) # read csv data

    # set directory to title
    path = './{}'.format(title)
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    # infer by, tot, and count columns
    if blabel is None:
        print('##No groupby given.')
        blabel = df.keys()[0]
        print('#\tInferred as \'{}\' from data.\n'.format(blabel))

    if tlabel is None:
        print('##No total column given.')
        tlabel = df.keys()[1]
        print('#\tInferred as \'{}\' from data\n'.format(tlabel))

    if clabel is None:
        print('##No count column given.')
        clabel = df.keys()[2]
        print('#\tInferred as \'{}\' from data\n'.format(clabel))

    p_vals = calculate_pvalues(df, blabel, tlabel, clabel, n, f=f, s=s, fname='p')
    q_vals = ana.calculate_qvalues(p_vals, s=s, fname='q')
