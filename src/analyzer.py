"""
Author: Joseph Min (kmin@caltech.edu)

This script is used to calculate the p and q values of a given dataset.
Bootstraps are multithreaded and numbaized for decreased computation time.
"""

import pandas as pd
import numpy as np
import itertools as it
import numba
import threading
import queue

def bootstrap_deltas(x, y, n, f=np.mean):
    """Given two datasets, return bootstrapped means.
    Params:
    x, y --- (list-like) datasets
    n    --- (int) # of bootstraps
    f    --- (function) to calculate deltas

    Returns:
    deltas -- a 1d array of length `n'
    """
    # get lengths
    nx = len(x)
    ny = len(y)

    # make a null dataset
    mixed = np.zeros(nx + ny)
    mixed[0:nx] = x
    mixed[nx:] = y

    # function to be numbaized --- bcs function that takes functions as
    # arguments can not be numbaized yet
    @numba.jit(nopython=True, nogil=True)
    def difference(x, y, n):
        """
        Calculates difference based on function f.
        """
        # initialize a delta vector to store everything in
        deltas = np.zeros(n)

        # go through the bootstrap

        # for each n
        for i in np.arange(n):
            # make new datasets that respect the null hypothesis that
            # mean(x) == mean(y) on average
            nullx = np.random.choice(mixed, nx, replace=True)
            nully = np.random.choice(mixed, ny, replace=True)

            # calculate the difference of their means
            delta = f(nully) - f(nullx)

            # store
            deltas[i] = delta

        return deltas

    deltas = difference(x, y, n)

    return deltas

def calculate_pvalues(df, by, which, n, f=np.mean, **kwargs):
    """
    Calculates the p values of each pairwise comparison.
    This function calls calculate_delta() and calculate_pvalue()

    Params:
    df    --- (pandas.DataFrame) data read from csv
    which --- (str) which column to perform comparison
    by    --- (String) label to group by (default: Genotype)
    n     --- (int) # of bootstraps
    f     --- (function) to calculate deltas

    kwargs:
    s     --- (boolean) whether to save matrix to csv (default: False)
    fname --- (str) csv file name
    ctrl  --- (str) control

    Returns:
    p_vals --- (pandas.DataFrame) of pairwise p values
    """
    # assign kwargs
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)
    ctrl = kwargs.pop('ctrl', None)

    df = df.set_index(by) # set genotype as index
    df = df.astype(float)

    # set_index('Genotype') must have already been done to matrix
    # matrix is a n x 2 matrix, column 1 for genotypes
    # and column 2 for measurments
    matrix = df.groupby(by)[which].unique()
    # now, matrix has been reduced to list of unique genotypes

    genotypes = matrix.keys() # list of all genotypes

    # 7/19/2017 unnecessary to save deltas
    # obs_delta = make_empty_dataframe(len(genotypes),\
    #         len(genotypes), genotypes, genotypes) # empty pandas dataframe
    # boot_deltas = make_empty_dataframe(len(genotypes),\
    #         len(genotypes), genotypes, genotypes) # empty pandas dataframe
    p_vals = make_empty_dataframe(len(genotypes),\
            len(genotypes), genotypes, genotypes) # empty pandas dataframe

    threads = []
    qu = queue.Queue()

    # for loop to iterate through all pairwise comparisons (not permutation)
    print('#Starting threads for bootstrapping...')
    # if no control is given, perform all pairwise comparisons
    if ctrl is None:
        for pair in it.combinations(genotypes, 2):

            thread = threading.Thread(target=calculate_deltas_queue,\
                                    args=(matrix, pair[0], pair[1], n, qu))
            threads.append(thread)

            thread.setDaemon(True)
            thread.start()

    # control given
    else:
        for genotype in genotypes:
            if genotype == ctrl:
                continue

            thread = threading.Thread(target=calculate_deltas_queue,
                                    args=(matrix, ctrl, genotype, n, qu))
            threads.append(thread)

            thread.setDaemon(True)
            thread.start()

    for thread in threads:
        gene_1, gene_2, delta_obs, deltas_bootstrapped = qu.get()
        p_vals[gene_1][gene_2] = calculate_pvalue(delta_obs, deltas_bootstrapped)

    print('#Bootstrapping of {} threads complete.\n'.format(len(threads)))

    print('#P-value matrix:')
    print(p_vals)

    # save matrix to csv
    if s:
        print('#Saving p-value matrix\n')
        save_matrix(p_vals, fname)

    return p_vals.astype(float)

def calculate_deltas_queue(matrix, gene_1, gene_2, n, queue, f=np.mean):
    """
    Function to calculate deltas with multithreading.
    Saves p values as tuples in queue.

    Params:
    matrix         --- (pandas.DataFrame) grouped data
    gene_1, gene_2 --- (String) genotypes to be compared
    n              --- (int) # of bootstraps
    queue          --- (queue.Queue) queue to save results
    f              --- (function) to calculate deltas

    Returns: none
    """
    delta_obs, deltas_bootstrapped = calculate_deltas(matrix[gene_1],\
                                                    matrix[gene_2], n)

    queue.put((gene_1, gene_2, delta_obs, deltas_bootstrapped))

def calculate_deltas(x, y, n, f=np.mean):
    """
    Calculates the observed and bootstrapped deltas.
    This function calls bootstrap_deltas()

    Params:
    x, y --- (list-like) observed values/measurements
    n    --- (int) # of bootstraps
    f    --- (function) to calculate deltas

    Returns:
    delta_obs           --- (float) observed delta
    deltas_bootstrapped --- (numpy.array) of bootstrapped deltas
    """
    delta_obs = y.mean() - x.mean()
    deltas_bootstrapped = bootstrap_deltas(x, y, n, f)

    return delta_obs, deltas_bootstrapped

@numba.jit(nopython=True, nogil=True)
def calculate_pvalue(delta, delta_array):
    """
    Calculates the pvalue of one observation.

    Params:
    delta       --- (float) the observed delta
    delta_array --- (list-like) boostrapped deltas

    Returns:
    p --- (float) p value
    """
    p = 0.0 # p value

    # sorted array to check if delta lies in the range
    # of bootstrapped deltas
    sorted_array = np.sort(delta_array)

    # if observed delta lies outside, p-value can not be directly computed
    # can only say p-value < 1/bootstraps
    # assign 0.0 for simpler computation - corrected later
    if delta < sorted_array[0] or delta > sorted_array[-1]:
        p = 0.0
    else:
        # number of points to the right of observed delta
        if delta >= 0:
            length = len(delta_array[delta_array >= delta])
            total_length = len(delta_array)
            p = length / total_length
        # number of points to the left of observed delta
        elif delta < 0:
            length = len(delta_array[delta_array <= delta])
            total_length = len(delta_array)
            p = length / total_length

    return p

def make_empty_dataframe(rows, cols, row_labels, col_labels):
    """
    Creates an empty dataframe with specified dimensions and
    labels.

    Params:
    rows       --- (int) # of rows
    cols       --- (int) # of columns
    row_labels --- (list) of row labels
    col_labels --- (list) of column labels

    Returns:
    matrix --- (pandas.DataFrame) wanted matrix
    """
    # make rows x cols zero dataframe
    matrix = pd.DataFrame(np.zeros((rows, cols)),\
                            dtype='object')
    # set column and row indices
    matrix.columns = list(col_labels)
    matrix = matrix.reindex(list(row_labels))

    return matrix

def calculate_qvalues(p_vals, **kwargs):
    """
    Calculates the q values.
    This function calls benjamin_hochberg_stepup()

    Params:
    p_vals --- (pandas.DataFrame) of pairwise p values
    kwargs:
    s     --- (boolean) whether to save matrix to csv (default: False)
    fname --- (str) output csv filename

    Returns:
    q_vals --- (pandas.DataFrame) of pairwise q values
    """
    # assign kwargs
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)

    # flatten 2d array to 1d
    flat = p_vals.values.flatten()

    q_vals_sorted, idx_no_nan = benjamin_hochberg_stepup(flat)

    q_vals = [np.nan] * len(flat)
    for index, q in zip(idx_no_nan, q_vals_sorted):
        q_vals[index] = q

    # reshape 1d array to dimensions of original 2d array
    q_vals = np.reshape(q_vals, p_vals.shape)

    # row & column labels
    labels = p_vals.index.values

    # construct pandas dataframe from data and labels
    q_vals = pd.DataFrame(data=q_vals, index=labels, columns=labels)

    print('#Q-value matrix:')
    print(q_vals)

    # save matrix to csv
    if s:
        print('#Saving q-value matrix\n')
        save_matrix(q_vals, fname)

    return q_vals

@numba.jit(nopython=True, nogil=True)
def benjamin_hochberg_stepup(p_vals):
    """
    Given a list of p-values, apply FDR correction and return the q values.

    Params:
    p_vals --- (list-like) p values (must be 1d)

    Returns:
    q_vals_sorted --- (np.array) q values
    idx_no_nan    --- (np.array) indices of q values
    """
    # sorted pvalues with respective original idices
    sort = np.sort(p_vals)
    idx = np.argsort(p_vals)

    # pvalues w/o nan values
    non_nan = ~np.isnan(sort)
    sort_no_nan = sort[non_nan]
    idx_no_nan = idx[non_nan]

    # empty list for qvalues
    q_vals_sorted = [np.nan] * len(sort_no_nan)
    prev_q = 0 # store previous q value

    # begin BH step up
    for i, p in enumerate(sort_no_nan):
        q = len(sort_no_nan) / (i+1) * p # calculate the q_value for the current point
        q = min(q, 1) # if q >1, make it == 1
        q = max(q, prev_q) # preserve monotonicity
        q_vals_sorted[i] = q # store q value
        prev_q = q # update the previous q value
    # end BH step up

    return q_vals_sorted, idx_no_nan

def get_signifcance(df, vals, control, by, which, threshold, f=np.mean):
    """
    Maps significance and statistics to dataframe.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    vals      --- (pandas.DataFrame) p/q values
    control   --- (str) control
    by        --- (str) index to group by
    which     --- (str) column to perform analysis
    threshold --- (float) p value threshold
    f         --- (function) to calculate statistic (default: np.mean)

    Returns:
    df2 --- (pandas.DataFrame) with mapped significance according to vals
    """
    df = df.copy() # need to operate on copy -- shouldn't change original matrix

    grouped = df.groupby(by) # dataframe grouped by index

    df_control = df[df[by] == control][which].values
    sig = {} # hash to store significance
    sig[control] = 'control'
    stat = {} # hash to store statistics

    for name, group in grouped:
        stat[name] = f(group[which])

        if not name == control:
            # assign correct p value (not nan)
            p = vals[control][name]
            if np.isnan(p):
                p = vals[name][control]

            if p < threshold:
                sig[name] = 'sig'
            else:
                sig[name] = 'non-sig'

    df['sig'] = df[by].map(sig)

    # sort by statistic
    df2 = df.copy()
    df2['stat'] = df2[by].map(stat)
    df2.sort_values('stat', inplace=True)

    return df2

def save_matrix(matrix, fname, replace=None):
    """
    Saves the matrix in both 2d and tidy csv format.

    Params:
    matrix  --- (pandas.DataFrame) matrix to be saved
    fname   --- (str) output csv filename
    replace --- (str) value to replace 0.0 values

    Returns: none
    """
    # replace values
    if not replace == None:
        matrix = matrix.replace(0, replace)

    matrix = matrix.dropna(how='all', axis=0)
    matrix = matrix.dropna(how='all', axis=1)

    matrix.to_csv('{}_matrix.csv'.format(fname))

    # convert to tidy format if matrix has more than one column
    if len(matrix.reset_index().keys()) > 2:
        tidy = matrix.reset_index()
        tidy.rename(columns={'index': 'col_1'}, inplace=True)
        tidy = pd.melt(tidy, id_vars='col_1', var_name="col_2", value_name="val")
        tidy = tidy.dropna()
        tidy.set_index('col_1', inplace=True) # prevent row index from being printed

        tidy.to_csv('{}_tidy.csv'.format(fname))


if __name__ == '__main__':
    import argparse
    import os

    n = 10**4
    stat = 'mean'
    fs = {'mean': np.mean,
            'median': np.median}

    parser = argparse.ArgumentParser(description='Run data anlysis \
                                        and plot boxplot.')
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
                        (defaults to first element of the csv file)',
                        type=str,
                        default=None)
    parser.add_argument('--save',
                        help='Save data to csv.',
                        action='store_true')
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    title = args.title
    n = args.b
    by = args.i
    s = args.save

    # change directory to title
    os.chdir(title)

    df = pd.read_csv(csv_path) # read csv data

    if by == None:
        by = df.keys()[0]

    # set directory to title
    path = './{}'.format(title)
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    # 7/18/2017 replaced
    # genotypes = np.unique(df_data['Genotype']) # get unique genotypes
    #
    # # begin counting samples
    # geno_counts = {}
    # for genotype in genotypes:
    #     geno_counts[genotype] = (df_data.Genotype == genotype).sum()
    # # end counting samples
    #
    # measurements = df_data.keys()[1:]

    # # calculate deltas
    # obs_deltas = calculate_deltas(obs_mean)
    # boot_deltas = calculate_deltas(boot_means)
    #
    # print(obs_deltas)
    # print(boot_deltas)
    #
    # print(obs_deltas.keys())

    for measurement in df:
        # don't check same column as group by
        if measurement == by:
            continue

        # calculate pvalues
        p_vals = calculate_pvalues(df, by, measurement, n, f=np.mean, save=s)
        p_vals = p_vals.astype(float)

        # calculate q_values
        q_vals = calculate_qvalues(p_vals, save=s)
        #
        # plot qvalues
        # plot_qvalue_heatmaps(q_vals, threshold, n, measurement, title=title)
        #
        # # plot boxplot -- plotting functionality moved to plotter.py
        # plot_boxplot(df_data, threshold, title=title)
