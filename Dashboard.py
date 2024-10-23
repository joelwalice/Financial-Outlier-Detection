import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
@st.cache
def load_data():
    df_raw = pd.read_csv('Dataset/creditcard.csv')
    df = df_raw.drop(['Time'], axis=1)
    return df

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def dbscan_outlier_detection(df, eps, min_samples):
    X = df.values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df['DBSCAN_Cluster'] = db.labels_
    df['DBSCAN_Outlier'] = (df['DBSCAN_Cluster'] == -1).astype(int)
    return df

def lof_outlier_detection(df, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_labels = lof.fit_predict(df)
    df['LOF_Score'] = -lof.negative_outlier_factor_
    df['LOF_Outlier'] = (outlier_labels == -1).astype(int)
    return df

def main():
    st.title("Outlier Detection Dashboard")

    df = load_data()
    st.write("### Raw Data", df.head())

    feature_list = df.columns.tolist()
    selected_features = st.multiselect("Select features for analysis:", feature_list, default=feature_list)

    if st.checkbox("Show histograms for selected features"):
        st.write("### Histograms")
        for feature in selected_features:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], bins=50, kde=True, ax=ax)
            ax.set_title(f"Histogram of {feature}")
            st.pyplot(fig)

    df_preprocessed = preprocess_data(df[selected_features])

    st.sidebar.title("DBSCAN Parameters")
    eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    min_samples = st.sidebar.slider("Minimum Samples", min_value=1, max_value=20, value=5, step=1)

    st.sidebar.title("LOF Parameters")
    n_neighbors = st.sidebar.slider("Neighbors (k)", min_value=1, max_value=50, value=20, step=1)

    df_with_dbscan = dbscan_outlier_detection(df_preprocessed.copy(), eps=eps, min_samples=min_samples)
    st.write("### DBSCAN Results")
    st.write(df_with_dbscan[['DBSCAN_Cluster', 'DBSCAN_Outlier']])

    df_with_lof = lof_outlier_detection(df_preprocessed.copy(), n_neighbors=n_neighbors)
    st.write("### LOF Results")
    st.write(df_with_lof[['LOF_Score', 'LOF_Outlier']])

    st.write("### DBSCAN vs LOF Outliers")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_with_dbscan[selected_features[0]],
        y=df_with_dbscan[selected_features[1]],
        hue=df_with_dbscan['DBSCAN_Outlier'],
        palette={0: 'blue', 1: 'red'},
        ax=ax,
        label="DBSCAN"
    )
    sns.scatterplot(
        x=df_with_lof[selected_features[0]],
        y=df_with_lof[selected_features[1]],
        hue=df_with_lof['LOF_Outlier'],
        palette={0: 'blue', 1: 'orange'},
        ax=ax,
        marker='X',
        label="LOF"
    )
    ax.set_title("Outliers Detected by DBSCAN and LOF")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
