"""

"""

import numpy as np
import pandas as pd
import numba
import expression_plotter_mt as mt

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib.dates as mdates

# causes runtime error for some reason
# rc('text', usetex=True)

rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14


def plot_qvalue_heatmap(df, by, which, threshold, n, f=np.mean, **kwargs):
    """
    Plots heatmap of q values. Saves graph.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    by        --- (String) which label to sort
    which     --- (str) which column to perform comparison
    threshold --- (float) q value threshold
    n         --- (int) # of bootstraps
    f=np.mean --- (function) to calculate deltas

    kwargs:
    title  --- (str) plot title
    xlabel --- (str) x label
    ylabel --- (str) y label
    p_vals --- (pandas.DataFrame) calculated p values
                    (automatically performs BH stepup algorithm to calculate qs)
    q_vals --- (pandas.DataFrame) calculated q values
                    (this function performs bootstraps if q_vals not given)
    """
    # organize kwargs
    title = kwargs.pop('title', 'title')
    xlabel = kwargs.pop('xlabel', 'xlabel')
    ylabel = kwargs.pop('ylabel', 'ylabel')

    # gate for kwarg arguments
    if 'q_vals' in kwargs:
        q_vals = kwargs.pop('q_vals')
    elif 'p_vals' in kwargs:
        p_vals = kwargs.pop('p_vals')
        q_vals = mt.calculate_qvalues(p_vals)

    ################################
    # begin plotting
    #
    # drop rows and columns with only nans
    q_vals = q_vals.dropna(how='all', axis=0)
    q_vals = q_vals.dropna(how='all', axis=1)

    # create mask for insiginificant & nan values
    array = q_vals.values.astype(float)
    mask = np.nan_to_num(array) > threshold
    mask_nan = np.isnan(array)

    # switch to current figure
    fig = plt.figure('heatmap')

    # convert values to reciprocal log
    values = q_vals.replace(0.0, 1/n)
    values = -values.apply(np.log10)
    vmax = -np.log10(1/n)
    vmin = -np.log10(threshold)


    # draw heatmap and apply mask
    ax = sns.heatmap(values, cmap='magma', mask=mask, vmin=vmin, vmax=vmax,\
                        cbar_kws={'label':r'$-\log_{10}(q)$'}, **kwargs)
    ax_mask = sns.heatmap(values.fillna(0),\
                            cmap=mpl.colors.ListedColormap(['white']),\
                            mask=~mask_nan, cbar=False)

    # figure settings
    ax.xaxis.tick_top()
    ax.invert_yaxis()

    fig.suptitle(title, y=1.07, fontsize=20)
    ax.set_ylabel(ylabel)
    plt.yticks(rotation='horizontal')
    plt.xticks(rotation=45)

    plt.plot()
    plt.savefig(title + '_' + which + '_heatmap.png', dpi=300, bbox_inches='tight')
    #
    # end plotting
    ###########################

# TODO: work in progress...
def plot_boxplot(df, by, which, threshold, n, f=np.mean, **kwargs):
    """
    Plot a boxplot.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    by        --- (str) index to group by
    which     --- (str) column to perform analysis
    threshold --- (float) p value threshold
    n         --- (int) # of bootstraps
    f         --- (function) to calculate delta (default: np.median)
    kwargs
    """

# TODO: does the jitterplot have to use q values too?
def plot_jitterplot(df, control, by, which, threshold, n, f=np.median, **kwargs):
    """
    Plot a stripplot ordered by the median value of each group.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    control   --- (str) control
    by        --- (str) index to group by
    which     --- (str) column to perform analysis
    threshold --- (float) p value threshold
    n         --- (int) # of bootstraps
    f         --- (function) to calculate delta (default: np.median)

    kwargs:
    title  --- (str) plot title
    xlabel --- (str) x label
    ylabel --- (str) y label
    p_vals --- (pandas.DataFrame) calculated p values
                    (this function performs bootstraps if p_vals not given)
    """
    title = kwargs.pop('title', 'title')
    xlabel = kwargs.pop('xlabel', 'xlabel')
    ylabel = kwargs.pop('ylabel', 'ylabel')

    # pop() causes the second argument to be calculated again for some reason,
    # even when 'p_vals' is in kwargs
    # p_vals = kwargs.pop('p_vals',
    #                         mt.calculate_pairwise_pvalues(df, by, which, n, f))

    # messy alternative to pop() bug(?)
    if 'p_vals' in kwargs:
        p_vals = kwargs.pop('p_vals')
    else:
        p_vals = mt.calculate_pairwise_pvalues(df, by, which, n, f)

    # begin data preparation
    grouped = df.groupby(by) # dataframe grouped by index

    df_control = df[df[by] == control][which].values
    ps = {} # hash to store p values
    ps[control] = 'control'
    stat = {} # hash to store statistics

    for name, group in grouped:
        stat[name] = f(group[which])

        if not name == control:
            # assign correct p value (not nan)
            p = p_vals[control][name]
            if np.isnan(p):
                p = p_vals[name][control]

            if p < threshold:
                ps[name] = 'sig'
            else:
                ps[name] = 'non-sig'

    df['sig'] = df[by].map(ps)

    # sort by median
    df2 = df.copy()
    df2['stat'] = df2[by].map(stat)
    df2.sort_values('stat', inplace=True)

    # plot figure
    fig = plt.figure('jitterplot')
    ax = sns.stripplot(x=which, y=by, data=df2, **kwargs)
    plt.axvline(df2[df2[by] == control][which].median(), ls='--', color='blue',\
                                                lw=1, label='control median')

    fig.suptitle(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    ax.yaxis.grid(False)

    plt.savefig(title + '_' + which + '_jitter.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    import argparse

    n = 100
    qval = 0.05
    stat = 'mean'
    fs = {'mean': np.mean,
            'median': np.median}

    parser = argparse.ArgumentParser(description='Plot graphs of significance.')
    # begin command line arguments
    parser.add_argument('csv_data',
                        help='The full path to the csv data file.',
                        type=str)
    parser.add_argument('type',
                        help='What kind of plot to generate. \
                        (heatmap, jitter, all)',
                        type=str,
                        choices=['heatmap', 'box', 'jitter', 'all'])
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
                        (default: {})'.format(qval),
                        type=float,
                        default=0.05)
    parser.add_argument('-i',
                        help='Label to group measurements by. \
                        (defaults to first element of the csv file)',
                        type=str,
                        default=None)
    parser.add_argument('-c',
                        help='Control genotype for jitterplot. \
                        (defaults to first genotype in csv file)',
                        type=str,
                        default=None)
    parser.add_argument('-s',
                        help='Statistic to perform bootstraps. \
                        (default: {})'.format(stat),
                        type=str,
                        choices=fs.keys(),
                        default='mean')
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    plot_type = args.type
    title = args.title
    n = args.b
    f = fs[stat]
    threshold = args.q
    by = args.i
    control = args.c

    df = pd.read_csv(csv_path) # read csv data

    if by == None:
        print('#No groupby argument given.')
        by = df.keys()[0]
        print('#\tInferred as \'{}\' from data.'.format(by))

    if control == None:
        print('#No control given.')
        control = df[by][0]
        print('#\tInferred as \'{}\' from data'.format(control))

    for measurement in df:
        if measurement == by:
            continue

        # calculate bootstraps
        p_vals = mt.calculate_pairwise_pvalues(df, by, measurement, n, f=f)
        p_vals = p_vals.astype(float)

        # TODO: spit out message if user selects jitterplot for > 100 datapoints
        # instead, recommend boxplot
        palette = {'sig': 'red', 'non-sig': 'black', 'control': 'blue'}
        if plot_type == 'heatmap':
            plot_qvalue_heatmap(df, by, measurement, threshold, n, f=f, p_vals=p_vals)
        elif plot_type == 'box':
            plot_boxplot(df, by, measurement, threshold, n, f=f)
        elif plot_type == 'jitter':
            plot_jitterplot(df, control, by, measurement, threshold, n, f=f,
                p_vals=p_vals, hue='sig', jitter=True, alpha=0.5, palette=palette)
        elif plot_type == 'all':
            # TODO: when user selects all plots, don't recalculate p value each time
            # just lookup p values calculated for heatmap
            plot_qvalue_heatmap(df, by, measurement, threshold, n, f=f, p_vals=p_vals)
            plot_jitterplot(df, control, by, measurement, threshold, n, f=f,
                p_vals=p_vals, hue='sig', jitter=True, alpha=0.5, palette=palette)
