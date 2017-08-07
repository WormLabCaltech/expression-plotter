import numpy as np
import pandas as pd
import analyzer as ana

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib.dates as mdates

class Simulation():
    """
    Class to simulate datasets and test analyzer.py
    """

    def __init__(self):
        """
        Initializer. Creates all test dataframes.
        """
        self.dfs = {}
        self.types = ['same', 'stat', 'diff']
        self.dfs['same'] = self.make_same_df()
        self.dfs['stat'] = self.make_stat_df()
        self.dfs['diff'] = self.make_diff_df()

    def simulate(self, df, n=10, boot_n=10**4, **kwargs):
        """
        Calculates p value n number of times.
        """
        results = []

        for i in range(n):
            blabel = kwargs.pop('blabel', 'name')
            mlabel = kwargs.pop('mlabel', 'measurement')

            p_vals = ana.calculate_pvalues(df, blabel, mlabel, boot_n)
            names = p_vals.index
            p_val = p_vals[names[0]][names[1]]

            results.append(p_val)

        return results

    def make_df(self, **kwargs):
        """
        Makes dataframe with given parameters.
        """
        cols = kwargs.pop('cols', ['col_1', 'col_2'])
        name = kwargs.pop('name', 'name')
        count = kwargs.pop('count', 30)
        dist = kwargs.pop('dist', lambda: np.random.normal(500, 100))
        vals = kwargs.pop('vals', None)

        # make data into array
        data = []
        for i in range(count):
            if vals is None:
                row = [name, dist()]
            else:
                row = [name, vals[i]]
            data.append(row)

        return self.convert_to_df(data, cols)


    def make_diff_df(self, **kwargs):
        """
        Makes a dataframe with different mean.
        """
        cols = kwargs.pop('cols', ['name', 'measurement'])
        names = kwargs.pop('names', ['wt', 'mt'])
        means = kwargs.pop('means', [500, 800])
        count = kwargs.pop('count', 30)

        dfs = []
        for name, mean in zip(names, means):
            df = self.make_df(cols=cols, name=name, count=count, dist=lambda: np.random.normal(mean, 100))
            dfs.append(df)

        df = pd.concat(dfs)
        return df

    def make_stat_df(self, **kwargs):
        """
        Makes a dataframe with the same mean.
        """
        cols = kwargs.pop('cols', ['name', 'measurement'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 30)
        dist = kwargs.pop('dist', lambda: np.random.normal(500, 100))

        dfs = []
        for name in names:
            df = self.make_df(cols=cols, name=name, count=count, dist=dist)
            dfs.append(df)

        df = pd.concat(dfs)
        return df


    def make_same_df(self, **kwargs):
        """
        Makes a dataframe with identical values.
        """
        cols = kwargs.pop('cols', ['name', 'measurement'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 30)
        dist = kwargs.pop('dist', lambda: np.random.normal(500, 100))

        vals = [dist() for i in range(count)]

        dfs = []
        for name in names:
            df = self.make_df(cols=cols, name=name, count=count, vals=vals)
            dfs.append(df)

        df = pd.concat(dfs)
        return df


    def convert_to_df(self, matrix, cols):
        """
        Converts matrix (list of lists) to dataframe.
        """
        df = pd.DataFrame(matrix, columns=cols)

        return df

if __name__ == '__main__':
    import os
    os.chdir('./simulation_output')

    n = 30
    boot_ns = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6]
    sim = Simulation()

    results = []
    for boot_n in boot_ns:
        novar_results = {}
        for t in sim.types:
            print('######## Simulating {} dataframe ########'.format(t))
            result = sim.simulate(sim.dfs[t], n=n, boot_n=boot_n)
            novar_results[t] = result

        results.append(novar_results)
        df = pd.DataFrame(novar_results)

        fig = plt.figure('jitter_{}'.format(boot_n))

        ax = sns.stripplot(data=df, jitter=True, alpha=0.5)
        ax.yaxis.grid(False)
        plt.savefig('jitter_{}.png'.format(boot_n), bbox_inches='tight')

    boot_results = {}
    for boot_n, result in zip(boot_ns, results):
        boot_results[str(boot_n)] = result

    df = pd.DataFrame(boot_results)
    for t in sim.types:
        plt.figure('jitter_{}'.format(t))
        ax = sns.stripplot(data=df.transpose()[t], jitter=True, alpha=0.5)
        ax.yaxis.grid(False)
        plt.savefig('jitter_{}.png'.format(t), bbox_inches='tight')
