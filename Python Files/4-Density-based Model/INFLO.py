def inflo_outlier_detection(df, features, k=20, percentile_threshold=0.75):
    X = df[features].values
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    local_densities = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        local_densities[i] = 1 / (np.mean(distances[i][1:]) + 1e-5)
    influences = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        influence = 0
        for j in indices[i][1:]:
            influence += local_densities[j]
        influences[i] = influence

    threshold = np.percentile(influences, percentile_threshold * 100)
    outlier_mask = influences < threshold
    df['INFLO_Score'] = influences
    df['Is_Outlier'] = outlier_mask.astype(int)

    return df
df_with_inflo = inflo_outlier_detection(df, feature_list)

print(df_with_inflo[['INFLO_Score', 'Is_Outlier']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_with_inflo[feature_list[0]], y=df_with_inflo[feature_list[1]], hue=df_with_inflo['Is_Outlier'])
plt.title('INFLO Outlier Detection Results')
plt.show()