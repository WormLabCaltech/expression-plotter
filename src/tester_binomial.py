"""
Tester script for analyzer_binary.py
"""
import numpy as np
import pandas as pd
import analyzer as ana
import analyzer_binary as bi

def make_test_dataframe(genes, ps, plate_n, total_n, columns, **kwargs):

    same = kwargs.pop('same', False)
    mean = kwargs.pop('mean', False)

    data = []
    for gene, p in zip(genes, ps):
        for i in range(plate_n):
            if gene is not genes[0] and same:
                count = data[i][2]
            elif gene is not genes[0] and mean and i == plate_n-1:
                total = np.sum([ele[2] for ele in data[:plate_n]])
                count = total - np.sum([ele[2] for ele in data[plate_n:]])
            else:
                count = np.random.binomial(total_n, p)
            row = [gene, total_n, count]
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    print(df)
    return df

if __name__ == '__main__':

    blabel = 'genotype'
    tlabel = 'total #'
    clabel = '#'

    columns = [blabel, tlabel, clabel]

    plate_n = 30 # number of plates for each genotype
    total_n = 100 # observations per plate
    p = 0.5 # p of wild type
    n = 10**5 # # of bootstraps
    test_n = 10**3 # of times to test

    # make dataframe
    print('#######Making test dataframes.#######')
    df_same = make_test_dataframe(['N2', 'same'], [0.5, 0.5], plate_n, total_n, columns, same=True)
    df_mean = make_test_dataframe(['N2', 'mean'], [0.5, 0.49], plate_n, total_n, columns, mean=True)
    # df_inf = make_test_dataframe(['N2', 'inf'], [0.1, 0.9], 10**3, 10**3, columns)

    big_ps = np.arange(0.0, 1.0, 0.1)
    df_big = make_test_dataframe([str(i) for i in big_ps], big_ps, plate_n, total_n, columns)
    df_big = df_big.set_index('genotype')
    df_big.to_csv('./test_data/big_test.csv')

    print('#######Checking p values#######')
    p_same = bi.calculate_pvalues(df_same, blabel, tlabel, clabel, n)
    p_mean = bi.calculate_pvalues(df_mean, blabel, tlabel, clabel, n)
    # p_inf = bi.calculate_pvalues(df_inf, blabel, tlabel, clabel, n)
    p_big = bi.calculate_pvalues(df_big, blabel, tlabel, clabel, n)
