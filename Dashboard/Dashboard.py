import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="Outlier Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    h1, h2, h3, h4, h5, h6 { color: #1F77B4; }
    .stButton>button { color: white; background-color: #1F77B4; }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.subheader("Feature Selection")
    features = st.sidebar.multiselect("Select Features", df.columns)

    st.sidebar.subheader("Outlier Detection Methods")
    methods = st.sidebar.multiselect("Choose Methods", ["IQR Method", "LOF", "INFLO", "DBSCAN", "ABOD", "LDOF"])

    st.title("Outlier Detection Dashboard")
    st.write("Visualize and detect outliers in your data using advanced algorithms.")
    st.write("### Dataset Preview")
    st.write(df.head())

    if "IQR Method" in methods:
        st.subheader("IQR Method")
        outlier_details, total_outliers = IQR_method(df, features)
        st.write(f"Total outliers across selected features: {total_outliers}")
        outlier_df = pd.DataFrame(outlier_details)
        st.dataframe(outlier_df)

    if "LOF" in methods:
        st.subheader("Local Outlier Factor (LOF)")
        contamination = st.sidebar.slider("LOF Contamination", 0.001, 0.05, 0.005, key="lof_contamination")
        lof_outliers = detect_outliers_lof(df, features, contamination=contamination)
        st.write("LOF Outliers detected at indices:", lof_outliers)

    if "INFLO" in methods:
        st.subheader("Influenced Outlierness Factor (INFLO)")
        k = st.sidebar.slider("Number of Neighbors (k)", 5, 50, 20, key="inflo_k")
        threshold = st.sidebar.slider("Outlier Threshold", 0.5, 1.0, 0.9, key="inflo_threshold")
        inflo_outliers = detect_outliers_inflo(df, features, k=k, threshold=threshold)
        st.write("INFLO Outliers detected at indices:", inflo_outliers)

    if "DBSCAN" in methods:
        st.subheader("DBSCAN")
        eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 1.0, 0.5, key="dbscan_eps")
        min_samples = st.sidebar.slider("Min Samples", 1, 10, 5, key="dbscan_min_samples")
        dbscan_outliers = detect_outliers_dbscan(df, features, eps=eps, min_samples=min_samples)
        st.write("DBSCAN Outliers detected at indices:", dbscan_outliers)

    if "ABOD" in methods:
        st.subheader("Angle-Based Outlier Detection (ABOD)")
        k = st.sidebar.slider("Number of Neighbors (k)", 5, 50, 5)
        abod_outliers = abod(df[features].values, k=k)
        st.write("ABOD Outliers detected at indices:", abod_outliers)

    if "LDOF" in methods:
        st.subheader("Local Density-based Outlier Factor (LDOF)")
        k = st.sidebar.slider("Number of Neighbors (k)", 5, 50, 20)
        ldof_outliers = compute_ldof(df[features].values, k=k)
        st.write("LDOF Outliers detected at indices:", ldof_outliers)

