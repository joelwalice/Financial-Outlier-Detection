#Static Bin Method

def static_bin_width_histogram_outliers(df, feature, bins=50, threshold=0.01):
    counts, bin_edges = np.histogram(df[feature], bins=bins)
    total_points = len(df[feature])

    densities = counts / total_points

    low_density_bins = np.where(densities < threshold)[0]

    lower_bound = bin_edges[low_density_bins.min()] if len(low_density_bins) > 0 else np.nan
    upper_bound = bin_edges[low_density_bins.max() + 1] if len(low_density_bins) > 0 else np.nan

    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=bins, kde=False, color='blue', label='Data')

    if not np.isnan(lower_bound):
        sns.histplot(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature], bins=bins, color='red', label='Outliers')
        plt.axvline(lower_bound, color='green', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')

    plt.title(f"Static Bin-Width Histogram of {feature} with Outliers")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

#Dynamic Bin Method

def dynamic_bin_width_histogram_outliers(df, feature, num_bins=50):
    sorted_values = np.sort(df[feature])
    total_points = len(sorted_values)

    points_per_bin = total_points // num_bins

    bin_edges = []
    densities = []

    for i in range(num_bins):
        bin_start = i * points_per_bin
        bin_end = (i + 1) * points_per_bin if (i + 1) * points_per_bin < total_points else total_points

        bin_edges.append(sorted_values[bin_start])
        densities.append(points_per_bin / (sorted_values[bin_end - 1] - sorted_values[bin_start] + 1e-10))

    bin_edges.append(sorted_values[-1])

    threshold = np.percentile(densities, 5)
    low_density_bins = np.where(np.array(densities) < threshold)[0]

    lower_bound = bin_edges[low_density_bins.min()] if len(low_density_bins) > 0 else np.nan
    upper_bound = bin_edges[low_density_bins.max() + 1] if len(low_density_bins) > 0 else np.nan

    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=num_bins, kde=False, color='blue', label='Data')

    if not np.isnan(lower_bound):
        sns.histplot(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature], bins=num_bins, color='red', label='Outliers')
        plt.axvline(lower_bound, color='green', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')

    plt.title(f"Dynamic Bin-Width Histogram of {feature} with Outliers")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

for feature in feature_list:
    static_bin_width_histogram_outliers(df, feature, bins=50, threshold=0.01)
    dynamic_bin_width_histogram_outliers(df, feature, num_bins=50)