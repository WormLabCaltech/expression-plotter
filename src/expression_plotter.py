"""

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
import matplotlib

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
def plot_qvalue_heatmaps(pvalues, threshold, **kwargs):
    """
    Plots the pvalues in separate heatmaps. (for each measurement)

    Params:
    pvalues   --- (dictionary) of pandas dataframes of p values
    threshold --- (float) of p value threshold

    Returns:

    """
    title = kwargs.pop('title', '')

    for i, measurement in enumerate(pvalues):
        print(pvalues[measurement])
        array = pvalues[measurement].values.astype(float)
        mask = np.nan_to_num(array) > threshold
        mask_nan = np.isnan(array)
        print(mask_nan)
        plt.figure(i)
        values = -pvalues[measurement].apply(np.log)
        bootstrap_n = 3 # TODO: replace this with non-hardcoded param
        values = values.replace(np.inf, bootstrap_n)
        vmax = bootstrap_n
        ax = sns.heatmap(-pvalues[measurement].apply(np.log), cmap='magma', mask=mask,\
                            vmin=0.0, vmax=vmax)
        sns.heatmap(-pvalues[measurement].fillna(0).apply(np.log), cmap=matplotlib.colors.ListedColormap(['white']), mask=~mask_nan,\
                            )

        ax.xaxis.tick_top()
        ax.invert_yaxis()
        plt.yticks(rotation='horizontal')
        plt.xticks(rotation=45)
        plt.plot()
        plt.savefig(title + '_' + measurement + '.svg')

def plot_boxplot(data, threshold, **kwargs):
    """
    """
    # data = data.groupby('Genotype')
    # plt.figure(10)
    # ax = sns.boxplot(data['exp1'])
    # plt.show()

def benjamin_hochberg_stepup(pvalues, bootstraps):
    """
    Given a list of p-values, apply FDR correction and return the q values.

    Params:
    pvalues --- (dictionary) of pandas dataframes of p values

    Returns:
    qvalues --- (dictionary) of pandas dataframes of q values
    """
    print(pvalues)
    qvalues = {}
    for measurement in pvalues:
        # flatten 2d array to 1d and convert uncertainties
        flat = pvalues[measurement].values.flatten()
        flat[flat == -1] = 1/bootstraps
        print(flat)

        # sorted pvalues with respective original idices
        sort = np.sort(flat)
        idx = np.argsort(flat)

        # pvalues w/o nan values
        no_nan = ~np.isnan(sort)
        sort_no_nan = sort[no_nan]
        idx_no_nan = idx[no_nan]

        print(sort_no_nan)
        print(idx_no_nan)

        # empty list for qvalues
        qvalues_sorted = [np.nan] * len(sort_no_nan)
        prev_q = 0 # store previous q value

        # begin BH step up
        for i, p in enumerate(sort_no_nan):
            q = len(sort_no_nan) / (i+1) * p # calculate the q_value for the current point
            q = min(q, 1) # if q >1, make it == 1
            q = max(q, prev_q) # preserve monotonicity
            qvalues_sorted[i] = q # store q value
            prev_q = q # update the previous q value
        # end BH step up
        print(qvalues_sorted)

        qvals = [np.nan] * len(sort)
        for index, q in zip(idx_no_nan, qvalues_sorted):
            qvals[index] = q

        # reshape 1d array to dimensions of original 2d array
        qvals = np.reshape(qvals, pvalues[measurement].shape)

        # row & column labels
        labels = pvalues[measurement].index.values

        # construct pandas dataframe from data and labels
        qvals = pd.DataFrame(data=qvals, index=labels, columns=labels)
        qvalues[measurement] = qvals

    return qvalues


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

    genotypes = np.unique(df_data['Genotype']) # get unique genotypes

    # begin counting samples
    geno_counts = {}
    for genotype in genotypes:
        geno_counts[genotype] = (df_data.Genotype == genotype).sum()
    # end counting samples

    measurements = df_data.keys()[1:]

    # calculate means
    obs_mean = calculate_stats(df_data) # obs_dev
    boot_means = calculate_means(df_data, geno_counts, bootstraps)

    # calculate deltas
    obs_deltas = calculate_deltas(obs_mean)
    boot_deltas = calculate_deltas(boot_means)

    print(obs_deltas)
    print(boot_deltas)

    print(obs_deltas.keys())

    # calculate pvalues
    pvalues = calculate_pvalues(boot_deltas, obs_deltas, bootstraps)

    # calculate q_values
    qvalues = benjamin_hochberg_stepup(pvalues, bootstraps)

    # plot qvalues
    plot_qvalue_heatmaps(qvalues, threshold, title=title)

    # plot boxplot
    plot_boxplot(df_data, threshold, title=title)
