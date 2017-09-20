import numpy as np
import pandas as pd
import analyzer_binary as ana

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

class SimulationBinary():
    """
    Class to simulate datasets and test analyzer_binary.py
    """
    def make_df(self, **kwargs):
        """
        Makes dataframe with given parameters.
        """
        cols = kwargs.pop('cols', ['col_1', 'col_2', 'col_3'])
        name = kwargs.pop('name', 'name')
        count = kwargs.pop('count', 50)
        total = kwargs.pop('total', 100)
        dist = kwargs.pop('dist', lambda: np.random.binomial(total, 0.5))
        vals = kwargs.pop('vals', None)

        # make data into array
        data = []
        for i in range(count):
            if vals is None:
                row = [name, dist(), total]
            else:
                row = [name, vals[i], total]
            data.append(row)

        return self.convert_to_df(data, cols)

    def convert_to_df(self, matrix, cols):
        """
        Converts matrix (list of lists) to dataframe.
        """
        df = pd.DataFrame(matrix, columns=cols)

        return df

    def make_same_df(self, **kwargs):
        """
        Makes a dataframe with identical values
        """
        cols = kwargs.pop('cols', ['name', 'count', 'total'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 50)
        total = kwargs.pop('total', 100)
        dist = kwargs.pop('dist', lambda: np.random.binomial(total, 0.5))

        vals = [dist() for i in range(count)]

        dfs = []
        for name in names:
            df = self.make_df(cols=cols, name=name, count=count, total=total, vals=vals)
            dfs.append(df)

        df = pd.concat(dfs)
        return df

    def make_stat_df(self, **kwargs):
        """
        Makes a dataframe with the same p.
        """
        cols = kwargs.pop('cols', ['name', 'count', 'total'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 50)
        total = kwargs.pop('total', 100)
        dist = kwargs.pop('dist', lambda: np.random.binomial(total, 0.5))

        dfs = []
        df = self.make_df(cols=cols, name=names[0], count=count, total=total, dist=dist)
        dfs.append(df)

        df = self.make_df(cols=cols, name=names[1], count=count, total=total, dist=dist)
        dfs.append(df)

        df = pd.concat(dfs)
        return df

    def make_diff_df(self, **kwargs):
        """
        Makes a dataframe with different p.
        """
        cols = kwargs.pop('cols', ['name', 'count', 'total'])
        names = kwargs.pop('names', ['wt', 'mt'])
        count = kwargs.pop('count', 50)
        total = kwargs.pop('total', 100)
        p = kwargs.pop('p', 0.7)

        dfs = []
        dist = lambda : np.random.binomial(total, 0.5)
        df = self.make_df(cols=cols, name=names[0], count=count, total=total, dist=dist)
        dfs.append(df)

        dist = lambda : np.random.binomial(total, p)
        df = self.make_df(cols=cols, name=names[1], count=count, total=total, dist=dist)
        dfs.append(df)

        df = pd.concat(dfs)
        return df

    def simulate1(self, n=10, boot_ns=[10**2, 10**3, 10**4, 10**5], **kwargs):
        """
        Simulates same dataframes.
        """
        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'count')
        tlabel = kwargs.pop('tlabel', 'total')

        dfs = {
            'same': self.make_same_df(),
            'stat': self.make_stat_df(),
            'diff': self.make_diff_df()
            }
        cols = ['type', 'n', 'p']
        results = []
        df_count = 0
        for key, df in dfs.items():
            df.to_csv('dfs/df_sim1_{}.csv'.format(key), index=False)
            df_count += 1
            for boot_n in boot_ns:
                for i in range(n):
                    progress = '{}/{} {}/{} {}/{}'.format(
                            df_count, len(dfs),
                            boot_ns.index(boot_n)+1, len(boot_ns),
                            i+1, n
                    )
                    print('sim1: {}'.format(progress))

                    p_vals = ana.calculate_pvalues(df, blabel, tlabel, mlabel, boot_n)
                    names = p_vals.index
                    p_val = p_vals[names[0]][names[1]]

                    results.append([key, boot_n, p_val])

        df = self.convert_to_df(results, cols)
        df.to_csv('sim1.csv', index=False)

    def simulate2(self, n=10, boot_ns=[10**2, 10**3, 10**4, 10**5], ps=np.arange(0, 1, 0.1), **kwargs):
        """
        Simulates varying dist p vs p value.
        """
        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'count')
        tlabel = kwargs.pop('tlabel', 'total')

        cols = ['n', 'dist', 'p']
        results = []
        for p in ps:
            df = self.make_diff_df(p=p)
            df.to_csv('dfs/df_sim2_{}.csv'.format(p), index=False)
            for boot_n in boot_ns:
                for i in range(n):
                    progress = '{}/{} {}/{} {}/{}'.format(
                            np.where(ps == p)[0][0], len(ps),
                            boot_ns.index(boot_n)+1, len(boot_ns),
                            i+1, n
                    )
                    print('sim2: {}'.format(progress))

                    p_vals = ana.calculate_pvalues(df, blabel, tlabel, mlabel, boot_n)
                    names = p_vals.index
                    p_val = p_vals[names[0]][names[1]]

                    results.append([boot_n, p, p_val])

        df = self.convert_to_df(results, cols)
        df.to_csv('sim2.csv', index=False)

    def simulate3(self, n=2, boot_n=10**5, ps=np.arange(0, 1, 0.01), **kwargs):
        """
        Simulates fisher exact vs bootstrapped p values.
        """
        # TODO: figure out how to test equality of proportions
        from scipy.stats import ttest_ind

        blabel = kwargs.pop('blabel', 'name')
        mlabel = kwargs.pop('mlabel', 'count')
        tlabel = kwargs.pop('tlabel', 'total')

        cols = ['dist', 'p', 't', 'p_t_2', 'p_t_1']
        results = []
        for p in ps:
            df = self.make_diff_df(p=p)
            df.to_csv('dfs/df_sim3_{}.csv'.format(p), index=False)
            for i in range(n):
                progress = '{}/{} {}/{}'.format(
                        np.where(ps == p)[0][0], len(ps),
                        i+1, n
                )
                print('sim3: {}'.format(progress))

                p_vals = ana.calculate_pvalues(df, blabel, tlabel, mlabel, boot_n)
                names = p_vals.index
                p_val = p_vals[names[0]][names[1]]

                # t test
                sample_1 = df[df[blabel] == names[0]]
                sample_2 = df[df[blabel] == names[1]]

                prop_1 = sample_1[mlabel] / sample_1[tlabel]
                prop_2 = sample_2[mlabel] / sample_2[tlabel]

                ttest = ttest_ind(prop_1, prop_2, equal_var=False)
                t = ttest[0]
                p_t = ttest[1]

                results.append([p, p_val, t, p_t, p_t/2])

        df = self.convert_to_df(results, cols)
        df.to_csv('sim3.csv', index=False)

if __name__ == '__main__':
    import os
    os.chdir('./simulation_output/binary')

    n = 50
    boot_ns = [10**2, 10**3, 10**4, 10**5]

    sim = SimulationBinary()
    # sim.simulate1(n=n, boot_ns=boot_ns)
    # sim.simulate2(n=n, boot_ns=boot_ns, ps=np.arange(0.4, 0.6, 0.01))
    sim.simulate3(ps=np.arange(0.4, 0.6, 0.001))



