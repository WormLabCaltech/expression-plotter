"""
TODO: simplify & organize code -- use **kwargs for functions with multiple args
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

# TODO: one-vs-all heatmap has very long axis -- need to fix
def plot_heatmap(df, by, which, threshold, n, f=np.mean, **kwargs):
    """
    Plots heatmap of q values. Saves graph.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    by        --- (String) which label to sort
    which     --- (str) which column to perform comparison
    threshold --- (float) q value threshold
    n         --- (int) # of bootstraps
    f         --- (function) to calculate deltas (default: np.mean)

    kwargs:
    s      --- (boolean) whether to save p/q value matrix (if not given) to csv
                        (default: False)
    fname  --- (str) output matrix filename
    title  --- (str) plot title
    xlabel --- (str) x label
    ylabel --- (str) y label
    vals --- (pandas.DataFrame) calculated p/q values
                    (performs bootstraps for q values if not given)
    """
    # organize kwargs
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)
    title = kwargs.pop('title', 'title')
    xlabel = kwargs.pop('xlabel', 'xlabel')
    ylabel = kwargs.pop('ylabel', 'ylabel')

    # gate for p/q values
    if 'vals' in kwargs:
        vals = kwargs.pop('vals')
    else:
        vals = mt.calculate_pvalues(df, by, which, n, f, s=s, fname=fname)
        vals = mt.calculate_qvalues(vals, s=s, fname=fname)

    ################################
    # begin plotting
    #
    # drop rows and columns with only nans
    vals = vals.dropna(how='all', axis=0)
    vals = vals.dropna(how='all', axis=1)

    # create mask for insiginificant & nan values
    array = vals.values.astype(float)
    mask = np.nan_to_num(array) > threshold
    mask_nan = np.isnan(array)

    # switch to current figure
    fig = plt.figure('heatmap')

    # convert values to reciprocal log
    values = vals.replace(0.0, 1/n)
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

    print('#Plotting heatmap')
    plt.savefig(which + '_heatmap.png', dpi=300, bbox_inches='tight')
    #
    # end plotting
    ###########################

# TODO: color-code significant boxes instead of using asterisks
def plot_boxplot(df, control, by, which, threshold, n, f=np.mean, **kwargs):
    """
    Plot a boxplot.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    control   --- (str) control
    by        --- (str) index to group by
    which     --- (str) column to perform analysis
    threshold --- (float) p value threshold
    n         --- (int) # of bootstraps
    f         --- (function) to calculate delta (default: np.mean)

    kwargs:
    s      --- (boolean) whether to save p/q matrix (if not given) to csv
                        (default: False)
    fname  --- (str) output matrix filename
    title  --- (str) plot title
    xlabel --- (str) x label
    ylabel --- (str) y label
    vals --- (pandas.DataFrame) calculated p/q values
                    (performs bootstraps for q values if not given)
    """
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)
    title = kwargs.pop('title', 'title')
    xlabel = kwargs.pop('xlabel', 'xlabel')
    ylabel = kwargs.pop('ylabel', 'ylabel')

    # gate for p/q values
    if 'vals' in kwargs:
        vals = kwargs.pop('vals')
    else:
        vals = mt.calculate_pvalues(df, by, which, n, f, s=s, fname=fname)
        vals = mt.calculate_qvalues(vals, s=s, fname=fname)

    fig = plt.figure('boxplot')
    ax = sns.boxplot(x=by, y=which, data=df, **kwargs)

    fig.suptitle(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # modify data to plot siginificance bars
    sig = vals < threshold
    sig = sig.reset_index()
    sig.rename(columns={'index': 'name_1'}, inplace=True)
    sig = pd.melt(sig, id_vars='name_1', var_name='name_2', value_name='sig')

    # plot significance bars
    sig_idxs = []
    sig_names = []
    lims = ax.get_ylim()
    offset = np.diff(lims)[0] * .05 # offset above max observation to start significance bar
    height = np.diff(lims)[0] * .1
    count = 0
    for i, row in sig.iterrows():
        if row['sig'] and control in row.values:
            idx_1 = np.where(p_vals.index == row['name_1'])[0][0]
            idx_2 = np.where(p_vals.index == row['name_2'])[0][0]

            y_1 = df[df[by] == row['name_1']].max()[which] + offset
            y_2 = df[df[by] == row['name_2']].max()[which] + offset
            x_1 = idx_1
            x_2 = idx_2

            y = df.max()[which] + height + (count * height)

            xs = [x_1, x_1, x_2, x_2]
            ys = [y_1, y, y, y_2]

            plt.plot(xs, ys, lw=1.5, c='k') # plot bars

            # calculate asterisk position
            x = (x_1 + x_2) / 2
            y = y - np.diff(lims) * .025
            plt.text(x, y, "*", ha='center', va='bottom', color='k', fontsize=16)
            count += 1

    print('#Plotting boxplot')
    plt.savefig(which + '_box.png', dpi=300, bbox_inches='tight')


def plot_jitterplot(df, control, by, which, threshold, n, f=np.mean, **kwargs):
    """
    Plot a stripplot ordered by the mean value of each group.

    Params:
    df        --- (pandas.DataFrame) data read from csv
    control   --- (str) control
    by        --- (str) index to group by
    which     --- (str) column to perform analysis
    threshold --- (float) p value threshold
    n         --- (int) # of bootstraps
    f         --- (function) to calculate delta (default: np.mean)

    kwargs:
    s      --- (boolean) whether to save p/q matrix (if not given) to csv
                        (default: False)
    fname  --- (str) output matrix filename
    title  --- (str) plot title
    xlabel --- (str) x label
    ylabel --- (str) y label
    vals   --- (pandas.DataFrame) calculated p/q values
                    (performs bootstraps for q values if not given)
    """
    s = kwargs.pop('s', False)
    fname = kwargs.pop('fname', None)
    title = kwargs.pop('title', 'title')
    xlabel = kwargs.pop('xlabel', 'xlabel')
    ylabel = kwargs.pop('ylabel', 'ylabel')

    # pop() causes the second argument to be calculated again for some reason,
    # even when 'p_vals' is in kwargs
    # p_vals = kwargs.pop('p_vals',
    #                         mt.calculate_pairwise_pvalues(df, by, which, n, f))

    # messy alternative to pop() bug(?)
    # gate for p/q values
    if 'vals' in kwargs:
        vals = kwargs.pop('vals')
    else:
        vals = mt.calculate_pvalues(df, by, which, n, f, s=s, fname=fname)
        vals = mt.calculate_qvalues(vals, s=s, fname=fname)

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
            p = vals[control][name]
            if np.isnan(p):
                p = vals[name][control]

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

    print('#Plotting jitterplot')
    plt.savefig(which + '_jitter.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    import argparse
    import os

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
                        help='Control genotype. \
                        (defaults to first genotype in csv file)',
                        type=str,
                        default=None)
    parser.add_argument('-m',
                        help='Statistic to perform bootstraps. \
                        (default: {})'.format(stat),
                        type=str,
                        choices=fs.keys(),
                        default='mean')
    parser.add_argument('--one-vs-all',
                        help='Perform one-vs-all analysis. (default: all-vs-all)',
                        action='store_true')
    parser.add_argument('--save',
                        help='Save data to csv. Plots are always saved regardless.',
                        action='store_true')
    # end command line arguments
    args = parser.parse_args()

    csv_path = args.csv_data
    plot_type = args.type
    title = args.title
    n = args.b
    f = fs[args.m]
    t = args.q
    by = args.i
    ctrl = args.c
    ova = args.one_vs_all
    s = args.save

    df = pd.read_csv(csv_path) # read csv data

    # set directory to title
    path = './{}'.format(title)
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    # infer groupby and controls
    if by == None:
        print('#No groupby argument given.')
        by = df.keys()[0]
        print('#\tInferred as \'{}\' from data.'.format(by))

    if ctrl == None:
        print('#No control given.')
        ctrl = df[by][0]
        print('#\tInferred as \'{}\' from data\n'.format(ctrl))

    for m in df:
        if m == by:
            continue

        if ova:
            ova_ctrl = ctrl
        else:
            ova_ctrl = None

        # calculate bootstraps
        p_vals = mt.calculate_pvalues(df, by, m, n, f=f, s=s,
                                            fname='{}_p'.format(m), ctrl=ova_ctrl)
        p_vals = p_vals.astype(float)
        q_vals = mt.calculate_qvalues(p_vals, s=s, fname='{}_q'.format(m))

        # arguments

        palette = {'sig': 'red',
                    'non-sig': 'black',
                    'control': 'blue'}
        jitter_kwargs = {'hue': 'sig',
                        'jitter': True,
                        'alpha': 0.5,
                        'palette': palette}


        if plot_type == 'heatmap':
            plot_heatmap(df, by, m, t, n, f=f, vals=q_vals, title=title)
        elif plot_type == 'box':
            plot_boxplot(df, ctrl, by, m, t, n, f=f, vals=q_vals, title=title)
        elif plot_type == 'jitter':
            # print warning if any measurement has >100 datapoints
            count = df.groupby(by)[m].size().max()
            if count > 100:
                print('\n#There are more than 100 observations for some measurements!')
                print('#Boxplots are recommended over jitterplots in favor of visibility.\n')

            plot_jitterplot(df, ctrl, by, m, t, n, f=f, vals=p_vals, title=title, **jitter_kwargs)
        elif plot_type == 'all':
            plot_heatmap(df, by, m, t, n, f=f, vals=p_vals, title=title)
            plot_boxplot(df, ctrl, by, m, t, n, f=f, vals=p_vals, title=title)
            plot_jitterplot(df, ctrl, by, m, t, n, f=f, vals=p_vals, title=title, **jitter_kwargs)
