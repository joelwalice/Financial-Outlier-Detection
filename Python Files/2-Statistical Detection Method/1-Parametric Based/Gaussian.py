# BOXPLOT

def IQR_method(df, n, features):

    outlier_list = []
    total_outliers = 0

    for column in features:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        outlier_list.extend(outlier_list_column)
        below_bound = df[df[column] < Q1 - outlier_step].shape[0]
        above_bound = df[df[column] > Q3 + outlier_step].shape[0]
        total_outliers += below_bound + above_bound
    outlier_count = Counter(outlier_list)
    multiple_outliers = [k for k, v in outlier_count.items() if v > n]
    print(f'Total number of outliers across all columns: {total_outliers}')

    return multiple_outliers

Outliers_IQR = IQR_method(df, 1, feature_list)