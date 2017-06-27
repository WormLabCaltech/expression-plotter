"""

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_stats(df_data):
    """
    Calculates the mean, standard dev of the sample.

    Params:
    df_data --- (pandas.DataFrame) data read from csv

    Returns:


    """
    means = df_data.groupby('Genotype').mean()
    #stdev = df_data.groupby('Genotype', as_index=False).std()

    return means.to_dict()



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
    means --- (dictionary) of lists of calculated means
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

    sns.distplot(boot_deltas['Fluorescence']['fog-2'])
    plt.gca().axvline(obs_delta['Fluorescence']['fog-2'])
    plt.show()

def calculate_deltas(means):
    """
    Calculates the deltas of means.

    Params:
    means --- (dictionary) of means (can be list of means)

    Returns:
    deltas --- (dictionary) of deltas
    """
    deltas = {}
    for measurement in means:
        deltas[measurement] = {}
        for genotype in means[measurement]:
            if genotype == 'WT':
                continue

            deltas[measurement][genotype] =\
                    means[measurement]['WT']\
                    - means[measurement][genotype]

    return deltas

def calculate_pvalues(boot_deltas, obs_deltas, bootstraps):
    """
    Calculates p values given the bootstrapped and observed deltas.

    Params:
    boot_deltas --- (dictionary) of bootstrapped deltas
    obs_deltas  --- (dictionary) of observed deltas

    Returns:
    pvalues --- (dictionary) of pvalues for each measurement
                            and mutant
    """
    # initialize hash
    pvalues = {}
    for measurement in obs_deltas:
        pvalues[measurement] = {}
        for genotype in obs_deltas[measurement]:
            # assign short variables for simple code
            delta = obs_deltas[measurement][genotype]
            delta_array = boot_deltas[measurement][genotype]
            print(delta)
            print(delta_array)

            # sorted array to check if delta lies in the range
            # of bootstrapped deltas
            sorted_array = np.sort(delta_array)

            # if observed delta lies outside, p-value can not be
            # directly computed.
            # can only say p-value < 1/bootstraps
            if delta < sorted_array[0] or delta > sorted_array[-1]:
                pvalues[measurement][genotype] = '<' +\
                                                    str(1/bootstraps)
                print('P-value for ' + genotype + ' could not be \
                        accurately calculated.')
            else:
                # number of points to the right of observed delta
                if delta > 0:
                    length = len(delta_array[delta_array >= delta])
                    total_length = len(delta_array)
                    pvalue = length / total_length
                    pvalues[measurement][genotype] = pvalue
                elif delta < 0:
                    length = len(delta_array[delta_array <= delta])
                    total_length = len(delta_array)
                    pvalue = length / total_length
                    pvalues[measurement][genotype] = pvalue

    print(pvalues)
    return pvalues




if __name__ == '__main__':
    import argparse

    bootstraps = 100

    parser = argparse.ArgumentParser(description='Run data anlysis \
                                        and plot boxplot.')
    # begin command line ArgumentParser
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
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    title = args.title
    bootstraps = args.b

    df_data = pd.read_csv(csv_path) # read csv data
    print(df_data)

    genotypes = np.unique(df_data['Genotype']) # get unique genotypes
    print(genotypes)

    # begin counting samples
    geno_counts = {}
    for genotype in genotypes:
        geno_counts[genotype] = (df_data.Genotype == genotype).sum()
    # end counting samples

    measurements = df_data.keys()[1:]
    print(measurements)

    # calculate means
    obs_mean = calculate_stats(df_data) # obs_dev
    boot_means = calculate_means(df_data, geno_counts, bootstraps)
    print(boot_means)

    # calculate deltas
    obs_deltas = calculate_deltas(obs_mean)
    boot_deltas = calculate_deltas(boot_means)

    print()
    print(boot_deltas)
    print()
    print(obs_deltas)

    # calculate pvalues
    pvalues = calculate_pvalues(boot_deltas, obs_deltas, bootstraps)



    #plot_bootstraps(boot_deltas, obs_deltas)


    #################
    # wt_mean = df_data[df_data.Genotype == 'WT']['Fluorescence'].mean()
    # mt_mean = df_data[df_data.Genotype == 'fog-2']['Fluorescence'].mean()
    # original_delta = mt_mean - wt_mean

    #plot_bootstraps(means, original_delta)
