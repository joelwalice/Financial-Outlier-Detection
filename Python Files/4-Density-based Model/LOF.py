def detect_outliers_lof(df, features, n_neighbors=30, contamination=0.005):
    X = df[features].values
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_
    df['LOF_Score'] = lof_scores
    df['Is_Outlier'] = (outlier_labels == -1).astype(int)

    return df

n_neighbors = 30
contamination = 0.005
df_with_lof = detect_outliers_lof(df, feature_list, n_neighbors, contamination)

print(df_with_lof[['LOF_Score', 'Is_Outlier']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_with_lof[feature_list[0]], y=df_with_lof[feature_list[1]], hue=df_with_lof['Is_Outlier'])
plt.title('Local Outlier Factor (LOF) Results with Adjusted Parameters')
plt.show()
