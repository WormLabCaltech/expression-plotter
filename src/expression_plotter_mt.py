"""

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
    delta -- a 1d array of length `n'
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
        delta = np.zeros(n)

        # go through the bootstrap
        # TODO: I'm fairly sure code below can be vectorized

        # for each n
        for i in np.arange(n):
            # make new datasets that respect the null hypothesis that
            # mean(x) == mean(y) on average
            nullx = np.random.choice(mixed, nx, replace=True)
            nully = np.random.choice(mixed, ny, replace=True)

            # calculate the difference of their means
            diff = f(nully) - f(nullx)

            # store
            delta[i] = diff

        return delta

    delta = difference(x, y, n)

    return delta

def calculate_pairwise_pvalues(df, by, which, n, f=np.mean):
    """
    Calculates the p values of each pairwise comparison.
    This function calls calculate_delta() and calculate_pvalue()

    Params:
    df    --- (pandas.DataFrame) data read from csv
    which --- (str) which column to perform comparison
    by    --- (String) label to group by (default: Genotype)
    n     --- (int) # of bootstraps
    f     --- (function) to calculate deltas

    Returns:
    p_vals --- (pandas.DataFrame) of pairwise p values
    """

    df = df.set_index(by) # set genotype as index
    df = df.astype(float)
    print(df)

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

    pairs = []
    threads = []
    qu = queue.Queue()

    # for loop to iterate through all pairwise comparisons (not permutation)
    for pair in it.combinations(genotypes, 2):
        # observed delta & bootstrapped deltas
        # delta, delta_array = calculate_delta(matrix[pair[0]], matrix[pair[1]],\
                                                #bootstraps)

        thread = threading.Thread(target=calculate_delta_queue,\
                                args=(matrix, pair[0], pair[1], n, qu))
        threads.append(thread)
        pairs.append(pair)

        print('Starting ' + str(thread))
        thread.setDaemon(True)
        thread.start()

        # assign to dataframe
        # TODO: is this assignment necessary? is it needed later?
        # obs_delta[pair[0]][pair[1]] = delta
        # boot_deltas[pair[0]][pair[1]] = delta_array
        #
        # # calculate p value
        # p_vals[pair[0]][pair[1]] = calculate_pvalue(delta, delta_array)


    # threads = [threading.Thread(target=calculate_delta, args=(matrix[pair[0]],\
    #                                 matrix[pair[1]], bootstraps, qu)) for pair in pairs]

    # for thread in threads:
    #     print('Starting ' + str(thread))
    #     thread.start()
    for thread in threads:
        gene_1, gene_2, delta_obs, deltas_bootstrapped = qu.get()
        p_vals[gene_1][gene_2] = calculate_pvalue(delta_obs, deltas_bootstrapped)

    return p_vals

def calculate_delta_queue(matrix, gene_1, gene_2, n, queue, f=np.mean):
    """
    Function to calculate deltas with multithreading.
    Saves p values as tuples in queue.

    Params:
    matrix         --- (pandas.DataFrame) data read from csv
    gene_1, gene_2 --- (String) genotypes to be compared
    n              --- (int) # of bootstraps
    queue          --- (queue.Queue) queue to save results
    f              --- (function) to calculate deltas

    Returns: none
    """
    delta_obs, deltas_bootstrapped = calculate_delta(matrix[gene_1],\
                                                    matrix[gene_2], n)

    queue.put((gene_1, gene_2, delta_obs, deltas_bootstrapped))

def calculate_delta(x, y, n, f=np.mean):
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

@numba.jit(nopython=True)
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
        if delta > 0:
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

def calculate_qvalues(p_vals):
    """
    Calculates the q values.
    This function calls benjamin_hochberg_stepup()

    Params:
    p_vals --- (pandas.DataFrame) of pairwise p values

    Returns:
    q_vals --- (pandas.DataFrame) of pairwise q values
    """
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

    return q_vals

@numba.jit(nopython=True)
def benjamin_hochberg_stepup(p_vals):
    """
    Given a list of p-values, apply FDR correction and return the q values.

    Params:
    p_vals --- (list-like) p values (must be 1d)

    Returns:
    q_vals --- (list-like) q values
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


if __name__ == '__main__':
    import argparse

    n = 100
    qval = 0.05

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
    parser.add_argument('-q',
                        help='Q value threshold for significance. \
                        (default: {0})'.format(qval),
                        type=float,
                        default=0.05)
    parser.add_argument('-i',
                        help='Label to group measurements by. \
                        (defaults to first element of the csv file)',
                        type=str,
                        default=None)
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    title = args.title
    n = args.b
    threshold = args.q
    by = args.i

    df = pd.read_csv(csv_path) # read csv data

    if by == None:
        by = df.keys()[0]

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
        p_vals = calculate_pairwise_pvalues(df, by, measurement, n, f=np.mean)
        p_vals = p_vals.astype(float)
        print(p_vals)

        # calculate q_values
        q_vals = calculate_qvalues(p_vals)
        print(q_vals)
        #
        # plot qvalues
        plot_qvalue_heatmaps(q_vals, threshold, n, measurement, title=title)
        #
        # # plot boxplot
        # plot_boxplot(df_data, threshold, title=title)
