import streamlit as st
import pandas as pd
from ray.tune import ExperimentAnalysis
import plotly.express as px
from pathlib import Path
import os

def load_data(path):
    analysis = ExperimentAnalysis(path)
    df = analysis.dataframe()
    return df

@st.cache_data
def load_ray_results_to_df(local_dir):
    dataframes = []
    for experiment in os.listdir(local_dir):
        experiment_dir = os.path.join(local_dir, experiment)
        for trial in os.listdir(experiment_dir):
            trial_dir = os.path.join(experiment_dir, trial)
            progress_path = os.path.join(trial_dir, "progress.csv")
            if os.path.exists(progress_path):
                df = pd.read_csv(progress_path)
                df["experiment_name"] = experiment
                df["trial_name"] = trial
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def main():
    st.title("Model Results")
    df = load_ray_results_to_df(Path("models/ray").resolve())
    if st.button('Show Raw Data'):
        st.write(df)
    st.markdown("## Performance Metrics")
    metrics = st.multiselect("Choose the metrics you want to view", df.columns.tolist(), default=df.columns.tolist())
    if st.button('Show Metrics'):
        st.line_chart(df[metrics])

if __name__ == "__main__":
    main()