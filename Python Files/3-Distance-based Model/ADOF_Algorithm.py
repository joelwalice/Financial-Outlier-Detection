def abod(data, k=5):
    n_samples = data.shape[0]
    abod_scores = np.zeros(n_samples)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(n_samples):
        neighbors = data[indices[i, 1:]]
        angles = []

        for j in range(len(neighbors)):
            for l in range(j + 1, len(neighbors)):
                vec1 = neighbors[j] - data[i]
                vec2 = neighbors[l] - data[i]
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                angles.append(angle)

        abod_scores[i] = np.var(angles) if angles else 0.0

    return abod_scores

data = df[feature_list].values
abod_scores = abod(data, k=5)

df['ABOD_Score'] = abod_scores
print(df[['ABOD_Score']])