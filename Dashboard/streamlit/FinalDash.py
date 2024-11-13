import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def IQR_method(df, features):
    outlier_details = []
    total_outliers = 0
    for column in features:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        lower_bound = Q1 - outlier_step
        upper_bound = Q3 + outlier_step

        below_bound = df[df[column] < lower_bound].shape[0]
        above_bound = df[df[column] > upper_bound].shape[0]
        total_column_outliers = below_bound + above_bound
        total_outliers += total_column_outliers

        outlier_percentage = (total_column_outliers / df.shape[0]) * 100
        outlier_details.append({
            "Feature": column,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers Below Bound": below_bound,
            "Outliers Above Bound": above_bound,
            "Total Outliers": total_column_outliers,
            "Outlier Percentage": round(outlier_percentage, 2)
        })
    return outlier_details, total_outliers

def stddev_method(df, features, threshold=3):
    outlier_counts = {}
    for column in features:
        mean = df[column].mean()
        std = df[column].std()
        outliers = df[(df[column] < mean - threshold * std) | (df[column] > mean + threshold * std)]
        outlier_counts[column] = len(outliers)
    return outlier_counts

def zscore_method(df, features, threshold=3):
    outlier_counts = {}
    for column in features:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = z_scores[abs(z_scores) > threshold]
        outlier_counts[column] = len(outliers)
    return outlier_counts

def modified_zscore_method(df, features, threshold=2.0):
    outlier_counts = {}
    for column in features:
        median = df[column].median()
        mad = (abs(df[column] - median)).median()
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        outliers = modified_z_scores[abs(modified_z_scores) > threshold]
        outlier_counts[column] = len(outliers)
    return outlier_counts

def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    st.pyplot(plt)

def detect_outliers_lof(df, features, contamination=0.005):
    X = df[features].values
    lof = LocalOutlierFactor(contamination=contamination)
    outlier_labels = lof.fit_predict(X)
    return np.where(outlier_labels == -1)[0]

def detect_outliers_inflo(df, features, k=20, threshold=0.75):
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

    threshold_value = np.percentile(influences, threshold * 100)
    outlier_mask = influences < threshold_value

    df['INFLO_Score'] = influences
    df['Is_Outlier_INFLO'] = outlier_mask.astype(int)
    return df

def detect_outliers_dbscan(df, features, eps=0.5, min_samples=5):
    X = df[features].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return np.where(labels == -1)[0]

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
    return np.argsort(-abod_scores)[:int(len(data) * 0.05)]

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

    return np.argsort(ldof_scores)[-int(len(ldof_scores) * 0.05):]

def load_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data["Return"] = stock_data["Close"].pct_change()
    return stock_data
    
def plot_outlier_scatter(df, method, outliers, feature1, feature2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=feature1, y=feature2, data=df,
        hue=['Outlier' if i in outliers else 'Normal' for i in range(len(df))],
        palette={'Outlier': 'red', 'Normal': 'blue'}
    )
    plt.title(f"Outlier Detection ({method})")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    st.pyplot(plt)

def pca_outliers(df, features, n_components=2, threshold=3):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[features])
    reconstruction = pca.inverse_transform(principal_components)
    reconstruction_error = np.sqrt(np.sum((df[features].values - reconstruction) ** 2, axis=1))
    outliers = np.where(reconstruction_error > threshold)[0]
    return outliers, reconstruction_error

def mahalanobis_outliers(df, features, threshold=3.0):
    mean_vector = np.mean(df[features], axis=0)
    covariance_matrix = np.cov(df[features], rowvar=False)
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    mahal_distances = df[features].apply(
        lambda row: mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)
    critical_value = chi2.ppf((1 - (1 / threshold)), len(features))
    outliers = mahal_distances[mahal_distances > critical_value].index
    return outliers, mahal_distances

def plot_outlier_distribution(outlier_scores, method_name):
    plt.figure(figsize=(8, 4))
    sns.histplot(outlier_scores, kde=True, color='purple')
    plt.title(f"{method_name} Outlier Score Distribution")
    plt.xlabel("Outlier Score")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def main():
    st.set_page_config(page_title="Enhanced Outlier Detection Dashboard", layout="wide")
    st.title("Enhanced Outlier Detection Dashboard")
    st.write("Upload a dataset and select outlier detection methods to visualize the results.")
    
    with st.sidebar:
        realtime = st.sidebar.checkbox("Need to do Real-time anomaly")
        st.header("Upload and Configure")
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            features = st.multiselect("Select Features for Outlier Detection", df.columns)
            methods = st.multiselect("Select Outlier Detection Methods", ["IQR Method", "Standard Deviation", "Z-Score", "LOF", "DBSCAN", "INFLO", "ABOD", "LDOF", "PCA Method", "Mahalanobis Distance"])

    if realtime:
        st.write("Real-time anomaly detection is enabled.")
        st.sidebar.header("Stock Selection")
        stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
        start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End Date", datetime.now())
        stock_data = load_stock_data(stock_symbol, start_date, end_date)
        st.subheader(f"Price data for {stock_symbol.upper()}")
        st.dataframe(stock_data.tail(), width=1200, height=200)

        st.subheader("Price History")
        st.line_chart(stock_data['Close'], width=0, height=400)

        def detect_anomalies_lof(df, features, contamination=0.01):
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            anomaly_results = {}
            for feature in features:
                df[f"{feature}_Anomaly"] = lof.fit_predict(df[[feature]])
                anomaly_results[feature] = df[df[f"{feature}_Anomaly"] == -1].index
            return anomaly_results

        features = ["Close", "Open", "High", "Low", "Volume"]
        anomalies = detect_anomalies_lof(stock_data, features)

        for feature, anomaly_indices in anomalies.items():
            st.subheader(f"{feature} Price with Detected Anomalies")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(stock_data.index, stock_data[feature], label=f"{feature} Price", color="#4CAF50")
            ax.scatter(anomaly_indices, stock_data.loc[anomaly_indices, feature], color='red', marker='o', label="Anomaly")
            ax.set_xlabel("Date", fontsize=12, color="#333333")
            ax.set_ylabel(f"{feature} Price", fontsize=12, color="#333333")
            ax.legend()
            st.pyplot(fig)

    if uploaded_file:
        st.subheader("Dataset Overview")
        st.write(df.head())
        
        with st.sidebar:
            st.write("### Customize Parameters")
            if "LOF" in methods:
                lof_contamination = st.slider("LOF Contamination", 0.001, 0.05, 0.005)
            if "DBSCAN" in methods:
                dbscan_eps = st.slider("DBSCAN Epsilon", 0.1, 1.0, 0.5)
                dbscan_min_samples = st.slider("DBSCAN Min Samples", 1, 10, 5)
            if "INFLO" in methods:
                inflo_k = st.slider("INFLO k", 5, 50, 20)
                inflo_threshold = st.slider("INFLO Threshold", 0.5, 1.0, 0.75)
            if "ABOD" in methods:
                abod_k = st.slider("ABOD k", 5, 50, 5)
            if "LDOF" in methods:
                ldof_k = st.slider("LDOF k", 5, 50, 5)
                
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["IQR Method", "Standard Deviation","Z-Score", "LOF", "DBSCAN", "INFLO", "ABOD", "LDOF", "PCA Method", "Mahalanobis Distance"])
        
        with tab1:
            if "IQR Method" in methods:
                st.subheader("IQR Method")
                outlier_details, total_outliers = IQR_method(df, features)
                st.write(f"Total Outliers Detected: {total_outliers}")
                st.write(outlier_details)
        with tab2:
            if "Standard Deviation" in methods:
                threshold = st.slider("Set threshold (Standard Deviations)", 1, 5, 3)
                outliers = stddev_method(df, features, threshold)
                st.write(f"Total Outliers Detected:")
                st.write(outliers)
        with tab3:
            if "Z-Score" in methods:
                threshold = st.slider("Set threshold (Z-Score)", 1, 5, 3)
                outliers = zscore_method(df, features, threshold)
                st.write(f"Total Outliers Detected using Z-Score:")
                st.write(outliers)
                threshold = st.slider("Set threshold (Modified Z-Score)", 3.0, 5.0, 3.5, step=0.1)
                updated_outliers = modified_zscore_method(df, features, threshold)
                st.write(f"Total Outliers Detected using Modified Z-Score: ")
                st.write(updated_outliers)
        with tab4:
            if "LOF" in methods:
                st.subheader("LOF Method")
                lof_outliers = detect_outliers_lof(df, features, lof_contamination)
                st.write(f"Outliers detected: {len(lof_outliers)}")
                plot_outlier_scatter(df, "LOF", lof_outliers, features[0], features[1])
                
        with tab5:
            if "DBSCAN" in methods:
                st.subheader("DBSCAN Method")
                dbscan_outliers = detect_outliers_dbscan(df, features, dbscan_eps, dbscan_min_samples)
                st.write(f"Outliers detected: {len(dbscan_outliers)}")
                plot_outlier_scatter(df, "DBSCAN", dbscan_outliers, features[0], features[1])

        with tab6:
            if "INFLO" in methods:
                st.subheader("INFLO Method")
                df_inflo = detect_outliers_inflo(df, features, k=inflo_k, threshold=inflo_threshold)
                st.write(f"Outliers detected: {df_inflo[df_inflo['Is_Outlier_INFLO'] == 1].shape[0]}")
                plot_outlier_distribution(df_inflo['INFLO_Score'], "INFLO")

        with tab7:
            if "ABOD" in methods:
                st.subheader("ABOD Method")
                abod_outliers = abod(df[features].values)
                st.write(f"Outliers detected: {len(abod_outliers)}")
                plot_outlier_scatter(df, "ABOD", abod_outliers, features[0], features[1])

        with tab8:
            if "LDOF" in methods:
                st.subheader("LDOF Method")
                ldof_outliers = compute_ldof(df[features].values, ldof_k)
                st.write(f"Outliers detected: {len(ldof_outliers)}")
                plot_outlier_distribution(ldof_outliers, "LDOF")
        with tab9:
            if "PCA Method" in methods:
                st.subheader("PCA Method")
                if len(features) > 1:
                    n_components = st.slider("Number of Components", 1, len(features), min(2, len(features)))
                    pca_threshold = st.slider("PCA Outlier Threshold", 1, 10, 3)
                    try:
                        pca_outliers_detected, pca_error = pca_outliers(df, features, n_components, pca_threshold)
                        st.write(f"Outliers detected using PCA: {len(pca_outliers_detected)}")
                        st.line_chart(pca_error, width=0, height=400)
                        plot_outlier_scatter(df, "PCA", pca_outliers_detected, features[0], features[1])
                    except Exception as e:
                        st.error(f"Error in PCA: {e}")
            else:
                st.warning("Select at least two features for PCA.")
        with tab10:
            if "Mahalanobis Distance" in methods:
                st.subheader("Mahalanobis Distance Method")
                maha_threshold = st.slider("Threshold for Mahalanobis Distance", 2.0, 5.0, 3.0)
                maha_outliers_detected, maha_distances = mahalanobis_outliers(df, features, maha_threshold)
                st.write(f"Outliers detected using Mahalanobis Distance: {len(maha_outliers_detected)}")
                st.line_chart(maha_distances, width=0, height=400)
                plot_outlier_scatter(df, "Mahalanobis Distance", maha_outliers_detected, features[0], features[1])
if __name__ == "__main__":
    main()
