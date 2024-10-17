k = int(np.sqrt(len(df)))

def knn_density(X, k):
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    densities = 1 / (np.mean(distances, axis=1) + 1e-5)
    return densities

def solving_set_outliers(df, features, k, threshold=0.01):
    X = df[features].values
    densities = knn_density(X, k)
    outlier_mask = densities < np.quantile(densities, threshold)
    return pd.Series(outlier_mask, index=df.index), densities

outliers, densities = solving_set_outliers(df, feature_list, k)

df['Density'] = densities
df['Is_Outlier'] = outliers

print(df[['Density', 'Is_Outlier']])

plt.figure(figsize=(10, 6))
sns.histplot(df['Density'], bins=50, kde=True)
plt.title('Density Estimation using Solving Set Approach')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[feature_list[0]], y=df[feature_list[1]], hue=df['Is_Outlier'], palette="coolwarm")
plt.title('Outliers based on Solving Set Approach')
plt.show()