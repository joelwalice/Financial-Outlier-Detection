def kde_sklearn(X, bandwidth):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(X)
    log_density = kde.score_samples(X)
    return np.exp(log_density)

def detect_outliers_kde(df, features, bandwidth, threshold=0.01):
    X = df[features].values
    densities = kde_sklearn(X, bandwidth)
    outlier_mask = densities < np.quantile(densities, threshold)
    return pd.Series(outlier_mask, index=df.index), densities

bandwidth = 1.0
outliers, densities = detect_outliers_kde(df, feature_list, bandwidth)

df['Density'] = densities
df['Is_Outlier'] = outliers

print(df[['Density', 'Is_Outlier']])

plt.figure(figsize=(10, 6))
sns.histplot(df['Density'], bins=50, kde=True)
plt.title('Density Estimation using Gaussian KDE')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()
