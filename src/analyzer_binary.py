import numpy as np
import pandas as pd
import numba
import multiprocessing as mp
import itertools as it
import analyzer as ana
import concurrent.futures as fut

def calculate_pvalues(df, blabel, tlabel, clabel, n, f=np.mean, **kwargs):
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

    # 8/1/2017 Replaced with processes
    # threads = []
    # qu = queue.Queue()

    cores = mp.cpu_count() # number of available cores

    # for loop to iterate through all pairwise comparisons (not permutation)
    # for loop to iterate through all pairwise comparisons (not permutation)
    print('#{} cores detected for this machine.'.format(cores))
    print('#Starting {} processes for bootstrapping...'.format(cores))
    with fut.ProcessPoolExecutor(max_workers=cores) as executor:
    # if no control is given, perform all pairwise comparisons
        if ctrl is None:
            fs = [executor.submit(calculate_deltas_process, matrix, tlabel, clabel,
                pair[0], pair[1], n) for pair in it.combinations(genotypes, 2)]

    # control given
        else:
            genotypes.remove(ctrl)
            fs = [executor.submit(calculate_deltas_process, matrix, tlabel, clabel,
                        ctrl, genotype, n) for genotype in genotypes]

        # save to matrix
        for f in fut.as_completed(fs):
            gene_1, gene_2, delta_obs, deltas_bootstrapped = f.result()
            p_vals[gene_1][gene_2] = ana.calculate_pvalue(delta_obs, deltas_bootstrapped)
    #     for pair in it.combinations(genotypes, 2):
    #
    #         thread = threading.Thread(target=calculate_deltas_queue,\
    #                                 args=(matrix, tlabel, clabel, pair[0], pair[1], n, qu))
    #         threads.append(thread)
    #
    #         thread.setDaemon(True)
    #         thread.start()
    #
    # # control given
    # else:
    #     for genotype in genotypes:
    #         if genotype == ctrl:
    #             continue
    #
    #         thread = threading.Thread(target=calculate_deltas_queue,
    #                                 args=(matrix, tlabel, clabel, ctrl, genotype, n, qu))
    #         threads.append(thread)
    #
    #         thread.setDaemon(True)
    #         thread.start()
    #
    # for thread in threads:
    #     gene_1, gene_2, delta_obs, deltas_bootstrapped = qu.get()
    #     p_vals[gene_1][gene_2] = ana.calculate_pvalue(delta_obs, deltas_bootstrapped)

    print('#Bootstrapping complete.\n')

    print('#P-value matrix:')
    print(p_vals)
    print()

    # save matrix to csv
    if s:
        print('#Saving p-value matrix\n')
        ana.save_matrix(p_vals, fname)

    return p_vals.astype(float)

def calculate_deltas_process(matrix, tlabel, clabel, gene_1, gene_2, n):
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

    # queue.put((gene_1, gene_2, delta_obs, deltas_bootstrapped))
    return gene_1, gene_2, delta_obs, deltas_bootstrapped

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
    stat_1 = f(cs_1 / ts_1)
    stat_2 = f(cs_2 / ts_2)
    delta_obs = stat_2 - stat_1

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
    # TODO: figure out best algorithm for speed

    @numba.jit(nopython=True, nogil=True)
    def calculate_stats(ts, p):
        l = len(ts)
        nullps = np.zeros(l)
        for i in np.arange(l):
            nullps[i] = np.random.binomial(ts[i], p) / ts[i]
        nullss = f(nullps)

        return nullss

    @numba.jit(nopython=True, nogil=True)
    def bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n):
        p = (np.sum(cs_1) + np.sum(cs_2)) / (np.sum(ts_1) + np.sum(ts_2))

        deltas = np.zeros(n)
        for i in np.arange(n):
            deltas[i] = calculate_stats(ts_2, p) - calculate_stats(ts_1, p)

        return deltas

    deltas = bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n)

    return deltas



    # @numba.jit(nopython=True, nogil=True)
    # def bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n):
    #     p = (np.sum(cs_1) + np.sum(cs_2)) / (np.sum(ts_1) + np.sum(ts_2))
    #
    #     deltas = np.zeros(n)
    #     for i in np.arange(n):
    #         # for each plate 1
    #         nullps_1 = np.zeros(len(ts_1))
    #         for j in np.arange(len(ts_1)):
    #             nullps_1[j] = np.random.binomial(ts_1[j], p) / ts_1[j]
    #         nullms_1 = np.mean(nullps_1)
    #
    #         # for each plate 2
    #         nullps_2 = np.zeros(len(ts_2))
    #         for j in np.arange(len(ts_2)):
    #             nullps_2[j] = np.random.binomial(ts_2[j], p) / ts_2[j]
    #         nullms_2 = np.mean(nullps_2)
    #
    #         deltas[i] = nullms_2 - nullms_1
    #
    #     return deltas

    # 8/1/2017 numba can't compile array expressions
    # def bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n):
    #     p = (np.sum(cs_1) + np.sum(cs_2)) / (np.sum(ts_1) + np.sum(ts_2))
    #     nullps_1 = np.zeros((len(ts_1), n))  # initialize blank array for sums
    #     # for each plate 1
    #     for i in np.arange(len(ts_1)):
    #         nullps_1[i,:] = np.random.binomial(ts_1[i], p, n) / ts_1[i]
    #     # find mean of plate 1
    #     nullms_1 = np.mean(nullps_1, axis=0)
    #
    #     nullps_2 = np.zeros((len(ts_2), n))  # initialize blank array for sums
    #     # for each plate 2
    #     for i in np.arange(len(ts_2)):
    #         nullps_2[i,:] = np.random.binomial(ts_2[i], p, n) / ts_2[i]
    #     # find mean of plate 2
    #     nullms_2 = np.mean(nullps_2, axis=0)
    #
    #     # find deltas
    #     deltas = nullms_2 - nullms_1
    #
    #     return deltas

    # 7/31/2017 This is a vectorized function, but numba does not support
    #           np.split and np.repeat
    # def bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n):
    #     # total probablity with labels removed
    #     p = (np.sum(cs_1) + np.sum(cs_2)) / (np.sum(ts_1) + np.sum(ts_2))
    #
    #     # vectorized bootstraps
    #     # make 2D array, each row representing plates, each column a bootstrap
    #     nullts_1 = np.split(np.repeat(ts_1, n), len(ts_1))
    #     # calculate binomial picks
    #     nullcs_1 = np.random.binomial(nullts_1, p)
    #     # calculate probability by dividing by total sample
    #     nullps_1 = nullcs_1 / ts_1[:,None]
    #     # calculate statistic using f
    #     nullss_1 = f(nullps_1, axis=0)
    #
    #     # make 2D array, each row representing plates, each column a bootstrap
    #     nullts_2 = np.split(np.repeat(ts_2, n), len(ts_2))
    #     # calculate binomial picks
    #     nullcs_2 = np.random.binomial(nullts_2, p)
    #     # calculate probability by dividing by total sample
    #     nullps_2 = nullcs_2 / ts_2[:,None]
    #     # calculate statistic using f
    #     nullss_2 = f(nullps_2, axis=0)
    #
    #     deltas = nullss_2 - nullss_1
    #
    #     return deltas

    # deltas = bootstrap_deltas_numba(ts_1, cs_1, ts_2, cs_2, n)
    #
    # return deltas



    # # 7/31/2017 vectorized by np.random.binomial
    # # total number of samples
    # ts_n = np.sum(ts_1) + np.sum(ts_2)
    # cs_n = np.sum(cs_1) + np.sum(cs_2)
    #
    # # mixed array
    # mixed = np.zeros(ts_n)
    # mixed[0:cs_n] = np.ones(cs_n)
    #
    # # function to be numbaized
    # @numba.jit(nopython=True, nogil=True)
    # def difference(ts_1, cs_1, ts_2, cs_2, n):
    #     """
    #     Calculates delta based on function f.
    #     """
    #
    #     # initialize deltas array
    #     deltas = np.zeros(n)
    #
    #     # perform bootstraps
    #     # TODO: use np.random.binomial - can it be done without looping n times?
    #     for i in np.arange(n):
    #         nullp_1 = np.zeros(len(ts_1))
    #         nullp_2 = np.zeros(len(ts_2))
    #
    #         for j in np.arange(len(ts_1)):
    #             nullc = np.sum(np.random.choice(mixed, cs_1[j], replace=True))
    #             nullp_1[j] = nullc / ts_1[j]
    #
    #         for j in np.arange(len(ts_2)):
    #             nullc = np.sum(np.random.choice(mixed, cs_2[j], replace=True))
    #             nullp_2[j] = nullc / ts_2[j]
    #
    #         # calculate difference of means
    #         delta = f(nullp_2) - f(nullp_1)
    #
    #         deltas[i] = delta
    #
    #     return deltas
    #
    # deltas = difference(ts_1, cs_1, ts_2, cs_2, n)
    #
    # return deltas

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
