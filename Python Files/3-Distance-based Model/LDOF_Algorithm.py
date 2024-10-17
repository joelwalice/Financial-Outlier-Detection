def compute_ldof(X, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)
    ldof_scores = []
    for i in range(len(X)):
        dist_to_neighbors = np.mean(distances[i])

        neighbors = X[indices[i]]
        neighbor_knn = NearestNeighbors(n_neighbors=k)
        neighbor_knn.fit(neighbors)
        neighbor_distances, _ = neighbor_knn.kneighbors(neighbors)
        avg_pairwise_dist = np.mean(neighbor_distances)

        ldof = dist_to_neighbors / avg_pairwise_dist if avg_pairwise_dist > 0 else 0
        ldof_scores.append(ldof)

    return np.array(ldof_scores)

def top_n_ldof(df, features, k, n):
    X = df[features].values
    ldof_scores = compute_ldof(X, k)
    df['LDOF_Score'] = ldof_scores
    top_n_outliers = df.nlargest(n, 'LDOF_Score')
    return top_n_outliers

k = 5
n = 10
top_n_outliers = top_n_ldof(df, feature_list, k, n)

print(top_n_outliers[['LDOF_Score']])