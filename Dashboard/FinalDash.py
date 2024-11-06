import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
import yfinance as yf
from datetime import datetime, timedelta

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

def plot_outlier_distribution(outlier_scores, method_name):
    plt.figure(figsize=(8, 4))
    sns.histplot(outlier_scores, kde=True, color='purple')
    plt.title(f"{method_name} Outlier Score Distribution")
    plt.xlabel("Outlier Score")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Main Streamlit App
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
            methods = st.multiselect("Select Outlier Detection Methods", ["IQR Method", "LOF", "DBSCAN", "INFLO", "ABOD", "LDOF"])

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
                
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["IQR Method", "LOF", "DBSCAN", "INFLO", "ABOD", "LDOF"])
        
        with tab1:
            if "IQR Method" in methods:
                st.subheader("IQR Method")
                outlier_details, total_outliers = IQR_method(df, features)
                st.write(f"Total Outliers Detected: {total_outliers}")
                st.write(outlier_details)

        with tab2:
            if "LOF" in methods:
                st.subheader("LOF Method")
                lof_outliers = detect_outliers_lof(df, features, lof_contamination)
                st.write(f"Outliers detected: {len(lof_outliers)}")
                plot_outlier_scatter(df, "LOF", lof_outliers, features[0], features[1])
                
        with tab3:
            if "DBSCAN" in methods:
                st.subheader("DBSCAN Method")
                dbscan_outliers = detect_outliers_dbscan(df, features, dbscan_eps, dbscan_min_samples)
                st.write(f"Outliers detected: {len(dbscan_outliers)}")
                plot_outlier_scatter(df, "DBSCAN", dbscan_outliers, features[0], features[1])

        with tab4:
            if "INFLO" in methods:
                st.subheader("INFLO Method")
                df_inflo = detect_outliers_inflo(df, features, k=inflo_k, threshold=inflo_threshold)
                st.write(f"Outliers detected: {df_inflo[df_inflo['Is_Outlier_INFLO'] == 1].shape[0]}")
                plot_outlier_distribution(df_inflo['INFLO_Score'], "INFLO")

        with tab5:
            if "ABOD" in methods:
                st.subheader("ABOD Method")
                abod_outliers = abod(df[features].values)
                st.write(f"Outliers detected: {len(abod_outliers)}")
                plot_outlier_scatter(df, "ABOD", abod_outliers, features[0], features[1])

        with tab6:
            if "LDOF" in methods:
                st.subheader("LDOF Method")
                ldof_outliers = compute_ldof(df[features].values, ldof_k)
                st.write(f"Outliers detected: {len(ldof_outliers)}")
                plot_outlier_distribution(ldof_outliers, "LDOF")

if __name__ == "__main__":
    main()
