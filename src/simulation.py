import numpy as np
import pandas as pd
import analyzer as ana

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib.dates as mdates

rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

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

    def simulate1_all(self, n=10, boot_ns=[10, 10**2, 10**3, 10**4, 10**5, 10**6], **kwargs):
        """
        Simulates for all dataframes.
        """
        self.sim1_results = {}
        self.sim1_results_l = []

        for t in self.types:
            self.sim1_results[t] = {}
            for boot_n in boot_ns:
                p_vals = self.simulate1(self.dfs[t], n=n, boot_n=boot_n)
                self.sim1_results[t][str(boot_n)] = p_vals

                for p in p_vals:
                    row = [t, boot_n, p]
                    self.sim1_results_l.append(row)


        # make into dataframe
        dfs = []
        for key, dic in self.sim1_results.items():
            df = pd.DataFrame(dic)
            df['key'] = key
            dfs.append(df)

        self.df_sim1 = pd.concat(dfs)
        self.df_sim1.reset_index(drop=True, inplace=True)

        self.df_sim1.to_csv('sim1.csv', index=False)

        self.df_sim1_l = pd.DataFrame(self.sim1_results_l, columns=['type', 'n', 'p'])
        self.df_sim1_l.to_csv('sim1_l.csv', index=False)

    def plot_sim1(self):
        """
        """
        for t in self.types:
            plt.figure('sim1_jitter_{}'.format(t))
            df = self.df_sim1[self.df_sim1['key'] == t]

            ax = sns.stripplot(data=df)
            plt.title('sim1_jitter_{}'.format(t))
            plt.xlabel('bootstraps')
            plt.ylabel('p')
            ax.yaxis.grid(False)

            plt.savefig('sim1_jitter_{}.svg'.format(t), bbox_inches='tight')

            # if some points are located close to each other, use -log scale
            ymax = df[df.keys()[:-1]].max().max()
            ymin = df[df.keys()[:-1]].min().min()
            cut = (ymax - ymin) * .1

            df_range = df[df.keys()[:-1]].max() - df[df.keys()[:-1]].min()
            df_log = self.df_sim1[(df_range[df_range < cut]).index]

            if not df_log.emtpy:
                for key in df_log:
                    n = int(key)
                    df_log[key][df_log[key] == 0.0] = 1/n
                df_log = df_log.apply(np.log10)

                plt.figure('sim1_jitter_{}_log'.format(t))
                ax = sns.stripplot(data=df_log)
                plt.title('sim1_jitter_{}_log'.format(t))
                plt.xlabel('bootstraps')
                plt.ylabel(r'$-\log_{10}(p)$')
                ax.yaxis.grid(False)

                plt.savefig('sim1_jitter_{}_log.svg'.format(t), bbox_inches='tight')

    def simulate1(self, df, n=10, boot_n=10**4, **kwargs):
        """
        Calculates p value n number of times.
        """
        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'measurement')

        results = []

        for i in range(n):
            p_vals = ana.calculate_pvalues(df, blabel, mlabel, boot_n)
            names = p_vals.index
            p_val = p_vals[names[0]][names[1]]

            results.append(p_val)

        return results

    def simulate2(self, n=10, boot_ns=[10, 10**2, 10**3], deltas=range(0,200,5), **kwargs):
        """
        Simulates varying delta vs p for different bootstraps.
        """
        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'measurement')

        self.sim2_results = []
        for boot_n in boot_ns:
            for delta in deltas:
                df = self.make_diff_df(mean=delta)

                for i in range(n):
                    print('{}/{} {}/{} {}/{}'.format(boot_ns.index(boot_n)+1, len(boot_ns), deltas.index(delta)+1, len(deltas), i+1, n))

                    p = ana.calculate_pvalues(df, blabel, mlabel, boot_n)
                    names = p.index
                    p = p[names[0]][names[1]]
                    row = [boot_n, delta, p]
                    self.sim2_results.append(row)

        # make into dataframe
        self.df_sim2 = pd.DataFrame(self.sim2_results, columns=['n', 'delta', 'p'])
        self.df_sim2.to_csv('sim2.csv', index=False)

    def plot_sim2(self):
        """
        Plots simulation 2 data.
        """
        plt.figure('sim2_scatter')
        ax = sns.lmplot(data=self.df_sim2, x='delta', y='p', hue='n')
        plt.title('sim2_scatter')

        plt.savefig('sim2_scatter.svg', bbox_inches='tight')


    def simulate3(self, boot_n=10**5, deltas=range(0,200), **kwargs):
        """
        Simulates t test vs bootstrapped p values.
        """
        from scipy.stats import ttest_ind

        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'measurement')

        self.sim3_results = []

        for delta in deltas:
            print('{}/{}'.format(deltas.index(delta), len(deltas)))
            df = self.make_diff_df(var=delta)
            p = ana.calculate_pvalues(df, blabel, mlabel, boot_n)
            names = p.index
            p = p[names[0]][names[1]]

            # t test
            sample_1 = df[df[blabel] == names[0]]
            sample_2 = df[df[blabel] == names[1]]
            ttest = ttest_ind(sample_1[mlabel], sample_2[mlabel], equal_var=False)
            t = ttest[0]
            p_t = ttest[1]

            row = [delta, p, t, p_t]
            self.sim3_results.append(row)

        self.df_sim3 = pd.DataFrame(self.sim3_results, columns=['variance', 'p', 't', 'p_t'])
        self.df_sim3.to_csv('sim3.csv', index=False)

    def plot_sim3(self):
        """
        Plots simulation 3 results.
        """
        plt.figure('sim3_scatter')
        ax = sns.lmplot(data=self.df_sim3, x='p_t', y='p')
        plt.title('sim3_scatter')

        plt.savefig('sim3_scatter.svg', bbox_inches='tight')

    def make_df(self, **kwargs):
        """
        Makes dataframe with given parameters.
        """
        cols = kwargs.pop('cols', ['col_1', 'col_2'])
        name = kwargs.pop('name', 'name')
        count = kwargs.pop('count', 50)
        dist = kwargs.pop('dist', lambda: np.random.normal(0, 100))
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
        count = kwargs.pop('count', 50)
        mean = kwargs.pop('mean', 50)
        var = kwargs.pop('var', 100)

        dfs = []
        df = self.make_df(cols=cols, name=names[0], count=count, dist=lambda: np.random.normal(0, 100))
        dfs.append(df)

        df = self.make_df(cols=cols, name=names[1], count=count, dist=lambda: np.random.normal(mean, var))
        dfs.append(df)

        df = pd.concat(dfs)
        return df

    def make_stat_df(self, **kwargs):
        """
        Makes a dataframe with the same mean.
        """
        cols = kwargs.pop('cols', ['name', 'measurement'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 50)
        var = kwargs.pop('var', 100)

        dfs = []
        df = self.make_df(cols=cols, name=names[0], count=count, dist=lambda: np.random.normal(0, 100))
        dfs.append(df)

        df = self.make_df(cols=cols, name=names[1], count=count, dist=lambda: np.random.normal(0, var))
        dfs.append(df)

        df = pd.concat(dfs)
        return df


    def make_same_df(self, **kwargs):
        """
        Makes a dataframe with identical values.
        """
        cols = kwargs.pop('cols', ['name', 'measurement'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 50)
        dist = kwargs.pop('dist', lambda: np.random.normal(0, 100))

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

    n = 50
    boot_ns = [10, 10**2, 10**3, 10**4, 10**5]
    sim = Simulation()

    sim.simulate1_all(n=n, boot_ns=boot_ns)
    sim.simulate2(n=10, boot_ns=boot_ns)
    sim.simulate3()

    # sim.plot_sim1()
    # sim.plot_sim2()
    # sim.plot_sim3()



    # results = []
    # for boot_n in boot_ns:
    #     novar_results = {}
    #     for t in sim.types:
    #         print('######## Simulating {} dataframe ########'.format(t))
    #         result = sim.simulate(sim.dfs[t], n=n, boot_n=boot_n)
    #         novar_results[t] = result
    #
    #     results.append(novar_results)
    #     df = pd.DataFrame(novar_results)
    #
    #     fig = plt.figure('jitter_{}'.format(boot_n))
    #
    #     ax = sns.stripplot(data=df, jitter=True, alpha=0.5)
    #     ax.yaxis.grid(False)
    #     plt.savefig('jitter_{}.png'.format(boot_n), bbox_inches='tight')
    #
    # boot_results = {}
    # for boot_n, result in zip(boot_ns, results):
    #     boot_results[str(boot_n)] = result
    #
    # df = pd.DataFrame(boot_results)
    # for t in sim.types:
    #     plt.figure('jitter_{}'.format(t))
    #     ax = sns.stripplot(data=df.transpose()[t], jitter=True, alpha=0.5)
    #     ax.yaxis.grid(False)
    #     plt.savefig('jitter_{}.png'.format(t), bbox_inches='tight')
