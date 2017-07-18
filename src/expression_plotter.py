"""

"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set(font_scale=1.5, font='Arial')
import matplotlib.pyplot as plt
import itertools as it
import matplotlib
import numba
import threading
import Queue

# TODO: not finished
def calculate_stats(df_data):
    """
    Calculates the mean, standard dev of the sample.

    Params:
    df_data --- (pandas.DataFrame) data read from csv

    Returns:

    """
    means = df_data.groupby('Genotype').mean()
    #stdev = df_data.groupby('Genotype', as_index=False).std()

    return means

@numba.jit(nopython=True)
def bootstrap_deltas(x, y, bootstraps):
    """Given two datasets, return bootstrapped means.
    Params:
    x, y --- datasets
    n ---number of iterations
    Output:
    delta -- a 1d array of length `n'
    """
    # get lengths
    nx = len(x)
    ny = len(y)

    # make a null dataset
    mixed = np.zeros(nx + ny)
    mixed[0:nx] = x
    mixed[nx:] = y

    # initialize a delta vector to store everything in
    delta = np.zeros(bootstraps)

    # go through the bootstrap
    # TODO: I'm fairly sure code below can be vectorized

    # for each n
    for i in np.arange(bootstraps):
        # make new datasets that respect the null hypothesis that
        # mean(x) == mean(y) on average
        nullx = np.random.choice(mixed, nx, replace=True)
        nully = np.random.choice(mixed, ny, replace=True)

        # calculate the difference of their means
        diff = nully.mean() - nullx.mean()

        # store
        delta[i] = diff

    return delta

def calculate_pairwise_pvalues(matrix, bootstraps):
    """
    Calculates the p values of each pairwise comparison.
    This function calls calculate_delta() and calculate_pvalue()

    Params:
    matrix     --- (pandas.DataFrame) data read from csv
    bootstraps --- (int) # of bootstraps

    Returns:
    p_vals --- (pandas.DataFrame) of pairwise p values
    """

    # set_index('Genotype') must have already been done to matrix
    # matrix is a n x 2 matrix, column 1 for genotypes
    # and column 2 for measurments
    matrix = matrix.groupby('Genotype').unique()
    # now, matrix has been reduced to list of unique genotypes

    genotypes = matrix.keys() # list of all genotypes

    obs_delta = make_empty_dataframe(len(genotypes),\
            len(genotypes), genotypes, genotypes) # empty pandas dataframe
    boot_deltas = make_empty_dataframe(len(genotypes),\
            len(genotypes), genotypes, genotypes) # empty pandas dataframe
    p_vals = make_empty_dataframe(len(genotypes),\
            len(genotypes), genotypes, genotypes) # empty pandas dataframe

    pairs = []
    # TODO: make thread list inside for loop & start them there
    # for loop to iterate through all pairwise comparisons (not permutation)
    for pair in it.combinations(genotypes, 2):
        # observed delta & bootstrapped deltas
        # delta, delta_array = calculate_delta(matrix[pair[0]], matrix[pair[1]],\
                                                #bootstraps)
        pairs.append(pair)

        # assign to dataframe
        # TODO: is this assignment necessary? is it needed later?
        obs_delta[pair[0]][pair[1]] = delta
        boot_deltas[pair[0]][pair[1]] = delta_array

        # calculate p value
        p_vals[pair[0]][pair[1]] = calculate_pvalue(delta, delta_array)

    queue = Queue.Queue()
    threads = [threading.Thread(target=calculate_delta, args=(matrix[pair[0]],\
                                    matrix[pair[1]], bootstraps)) for pair in pairs]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for i in range(queue.qsize()):
        delta, delta_array = queue.get()

    return p_vals

@numba.jit(nopython=True)
def calculate_delta(x, y, bootstraps, queue):
    """
    Calculates the observed and bootstrapped deltas.
    This function calls bootstrap_deltas()

    Params:
    x, y       --- (list-like) observed values/measurements
    bootstraps --- (int) # of bootstraps

    Returns:
    delta_obs           --- (float) observed delta
    deltas_bootstrapped --- (numpy.array) of bootstrapped deltas
    """
    delta_obs = y.mean() - x.mean()
    deltas_bootstrapped = bootstrap_deltas(x, y, bootstraps)
    queue.put((delta_obs, deltas_bootstrapped))

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

# TODO: no longer needed
def calculate_means(df_data, geno_counts, bootstraps):
    """
    Calculates the means of each measurement.
    Each array in data is randomly sampled equal amount of times
    as specified in the genotype geno_counts hash.
    Returns dictionary of bootstrapped means with measurements as
    keys.

    Params:
    data        --- (pandas.DataFrame) data read from csv
    geno_counts --- (dictionary) of number of genotypes
    bootstraps  --- (int) number of bootstraps

    Returns:
    means --- (dictionary) of pandas dataframes of means
    """
    # extract data w/o genotype lables
    # column names are the keys to the dictionary
    matrix = df_data[df_data.columns[1:]].to_dict(orient='series')

    # initialize hash
    means = {}
    for measurement in matrix:
        means[measurement] = {}
        for genotype in geno_counts:
            means[measurement][genotype] = np.zeros(bootstraps)

    # calculate bootstraps
    for i in range(bootstraps):
        for measurement in matrix:
            for genotype in geno_counts:
                count = geno_counts[genotype]
                # sample with replacement the number of genotypes
                sample = matrix[measurement].sample(count,
                                                    replace=True)
                mean = sample.mean()
                means[measurement][genotype][i] = mean

    return means

# TODO: no longer needed
def plot_bootstraps(boot_deltas, obs_delta):
    """
    """

    f, axarr = plt.subplots(len(boot_deltas), sharex=True)
    for i, measurement in enumerate(boot_deltas):
        legend = []
        for genotype in boot_deltas[measurement]:
            sns.distplot(boot_deltas[measurement][genotype],\
                            ax=axarr[i], label=genotype)
            legend.append(genotype)
            # color = axarr[i].gca().get_color()
            axarr[i].axvline(obs_delta[measurement][genotype])
        axarr[i].legend()
    plt.show()

# TODO: no longer needed
def calculate_deltas(means):
    """
    Calculates the deltas of means.

    Params:
    means --- (dictionary) of means (can be list of means)

    Returns:
    deltas --- (dictionary) of pandas dataframe of deltas
    """
    deltas = {}
    for measurement in means:
        genotypes = means[measurement].keys()

        # make empty dataframe
        matrix = make_empty_dataframe(len(genotypes),\
                len(genotypes), genotypes, genotypes)

        # iterate through each pairwise combination
        for pair in it.combinations(genotypes, 2):
            delta = means[measurement][pair[0]]\
                    - means[measurement][pair[1]]
            matrix[pair[0]][pair[1]] = delta

        # assign matrix to hash
        deltas[measurement] = matrix

    return deltas

# TODO: no longer needed
def calculate_pvalues(boot_deltas, obs_deltas, bootstraps):
    """
    Calculates p values given the bootstrapped and observed deltas.

    Params:
    boot_deltas --- (dictionary) of bootstrapped deltas
    obs_deltas  --- (dictionary) of observed deltas

    Returns:
    pvalues --- (dictionary) of pandas dataframes of p values
    """
    # initialize hash
    pvalues = {}
    for measurement in obs_deltas:
        genotypes = obs_deltas[measurement].keys()

        # make empty dataframe
        matrix = make_empty_dataframe(len(genotypes),\
                    len(genotypes), genotypes, genotypes)

        # iterate through each pairwise combination
        for pair in it.combinations(genotypes, 2):
            # assign short variables for simple code
            delta = obs_deltas[measurement][pair[0]][pair[1]]
            boot_delta_array = boot_deltas[measurement]\
                                [pair[0]][pair[1]]

            # sorted array to check if delta lies in the range
            # of bootstrapped deltas
            sorted_array = np.sort(boot_delta_array)

            # if observed delta lies outside, p-value can not be
            # directly computed.
            # can only say p-value < 1/bootstraps
            if delta < sorted_array[0] or delta > sorted_array[-1]:
                matrix[pair[0]][pair[1]] = 0.0
                print('P-value for ' + measurement + ' ' + pair[0] + ' vs '\
                            + pair[1] + ' could not be accurately calculated.')
            else:
                # number of points to the right of observed delta
                if delta > 0:
                    length = len(boot_delta_array[boot_delta_array\
                                    >= delta])
                    total_length = len(boot_delta_array)
                    pvalue = length / total_length
                    matrix[pair[0]][pair[1]] = pvalue
                elif delta < 0:
                    length = len(boot_delta_array[boot_delta_array\
                                    <= delta])
                    total_length = len(boot_delta_array)
                    pvalue = length / total_length
                    matrix[pair[0]][pair[1]] = pvalue

        # assign matrix to hash
        matrix = matrix.astype(float)
        pvalues[measurement] = matrix
    return pvalues

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

# TODO: work in progress...
def plot_qvalue_heatmaps(q_vals, threshold, bootstraps, figure, **kwargs):
    """
    Plots the p values in separate heatmaps. (for each measurement)

    Params:
    q_vals     --- (pandas.DataFrame) of q values
    threshold  --- (float) of q value threshold
    bootstraps --- (int) # of bootstraps
    figure     --- (String) figure title (default: measurment)

    Returns:

    """
    # title to be appended to filename
    title = kwargs.pop('title', '')

    # drop rows and columns with only nans
    q_vals = q_vals.dropna(how='all', axis=0)
    q_vals = q_vals.dropna(how='all', axis=1)

    # create mask for insiginificant & nan values
    array = q_vals.values.astype(float)
    mask = np.nan_to_num(array) > threshold
    mask_nan = np.isnan(array)

    # switch to current figure
    fig = plt.figure(figure)

    # convert values to reciprocal log
    values = q_vals.replace(0.0, 1/bootstraps)
    values = -values.apply(np.log10)
    vmax = -np.log10(1/bootstraps)
    vmin = -np.log10(threshold)
    print(values)


    # draw heatmap and apply mask
    ax = sns.heatmap(values, cmap='magma', mask=mask, vmin=vmin, vmax=vmax,\
                        cbar_kws={'label':r'$-\log_{10}$'})
    ax_mask = sns.heatmap(values.fillna(0),\
                            cmap=matplotlib.colors.ListedColormap(['white']),\
                            mask=~mask_nan, cbar=False)

    # figure settings
    hfont = {'fontname': 'Pragmatica Light'}
    ax.xaxis.tick_top()
    ax.invert_yaxis()

    fig.suptitle(figure, y=1.05, fontsize=20, fontname='Arial')
    ax.set_ylabel('Genotypes', x=1.05, fontsize=16, fontname='Arial')
    plt.yticks(rotation='horizontal', fontsize=14, fontname='Arial')
    plt.xticks(rotation=45, fontsize=14, fontname='Arial')



    plt.plot()
    plt.savefig(title + '_' + measurement + '.png', dpi=300, bbox_inches='tight')

# TODO: work i progress...
def plot_boxplot(data, threshold, **kwargs):
    """
    """
    # data = data.groupby('Genotype')
    # plt.figure(10)
    # ax = sns.boxplot(data['exp1'])
    # plt.show()

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

    bootstraps = 100
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
                        (default: {0})'.format(bootstraps),
                        type=int,
                        default=100)
    parser.add_argument('-q',
                        help='Q value threshold for significance.\
                        (default: {0})'.format(qval),
                        type=float,
                        default=0.05)
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    title = args.title
    bootstraps = args.b
    threshold = args.q

    df_data = pd.read_csv(csv_path) # read csv data
    matrix = df_data.set_index('Genotype') # set genotype as index
    matrix = matrix.astype(float)

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

    for measurement in matrix:
        # calculate pvalues
        p_vals = calculate_pairwise_pvalues(matrix[measurement], bootstraps)
        p_vals = p_vals.astype(float)
        print(p_vals)

        # calculate q_values
        q_vals = calculate_qvalues(p_vals)
        print(q_vals)
        #
        # plot qvalues
        plot_qvalue_heatmaps(q_vals, threshold, bootstraps, measurement, title=title)
        #
        # # plot boxplot
        # plot_boxplot(df_data, threshold, title=title)
