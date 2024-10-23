X = df[feature_list].values

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = dbscan.fit_predict(X)

df['Is_Outlier'] = (df['Cluster'] == -1).astype(int)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['Is_Outlier'] = df['Is_Outlier']

print(df[['Cluster', 'Is_Outlier']].head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Is_Outlier', data=df_pca, palette='coolwarm')
plt.title('DBSCAN Outlier Detection with PCA')
plt.show()