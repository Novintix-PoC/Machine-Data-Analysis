import os
import io
import numpy as np
import streamlit as st
import plotly.express as px
import time
import pandas as pd
from datetime import datetime
from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv

st.set_page_config(page_title="Machine Dashboard", layout="wide")
# Load Azure credentials
# load_dotenv()
account_name = os.getenv('ACC_NAME')
account_key = os.getenv('ACC_KEY')
container_name = 'machinedatademo'

# Connect to Azure Blob Storage
connect_str = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# Function to load .npy file directly from blob into numpy array
def load_npy_from_blob(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob()
    npy_bytes = io.BytesIO(blob_data.readall())
    return np.load(npy_bytes, allow_pickle=True)

def load_csv_from_blob(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob()
    return pd.read_csv(io.BytesIO(blob_data.readall()))

# Load .npy files directly into memory from blob
start_time = datetime.now()
data = load_npy_from_blob('data.npy')
duration = datetime.now() - start_time
print(f'Data loaded in {duration.seconds // 3600}h {duration.seconds % 3600 // 60}m {duration.seconds % 60}s.')
predictions = load_npy_from_blob('predictions.npy')

state_transition_data = load_npy_from_blob('next_state_steps.npy')
# Extract classes
classes = data[:, -1].astype(int)

predicted_steps = state_transition_data[:,1]  # row 1 corresponds to 'predicted_steps'
predicted_next_state = state_transition_data[:,0] # row 6 corresponds to 'predicted_next_state'

mapping_df = load_csv_from_blob('mapping.csv')
# Create a dictionary mapping column index to parameter name
param_mapping = dict(zip(mapping_df['index'], mapping_df['_NAME']))

valid_classes = [0, 1, 2, 3]
classes = np.array([c if c in valid_classes else 0 for c in classes])

# Simple sanity check to ensure no out-of-bounds
n_data_rows = data.shape[0]
n_pred_rows = predictions.shape[0]
n_rows = min(n_data_rows, n_pred_rows)  # We'll only display up to this many rows
 
# A class label mapping for readability
label_mapping = {
    0: "Active",
    1: "Idle",
    2: "Spindle Warm-up",
    3: "Alarm"
}
 
# Apply custom styling
def apply_custom_styling():
    st.markdown("""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        /* Main app styling */
        .stApp {
            background-color: rgb(240, 240, 240);
            font-family: 'Roboto', sans-serif;
        }
                
        [data-testid="stHeader"]{color:#000000;}
                
        /* Header styling */
        .dashboard-header {
            background-color: #FFA500;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #000;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .dashboard-header .logo {
            font-weight: 700;
            font-size: 24px;
        }
        .dashboard-header .title {
            font-size: 24px;
            font-weight: Bold;
        }
        /* Status panels styling */
        .status-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 20px;
            background-color: #fff;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .status-box {
            flex: 1;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .status-box .label {
            font-size: 20px;
            color:#000000;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .status-box .value {
            font-size: 20px;
            color: #000000;
            font-weight: 700;
        }
        .active-status {
            color: #008000 !Important;
        }
        .idle-status {
            color: #FFD700 !Important;
        }
        /* Probability panels styling */
        .probability-title {
            font-size: 25px;
            font-weight: bold;
            color:#000000;
            margin-bottom: 10px;
        }
        .probability-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 20px;
        }
        .probability-box {
            flex: 1;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
        }
        .probability-box .label {
            font-size: 18px;
            font-weight: Bold;
            margin-bottom: 5px;
        }
        .probability-box .value {
            font-size: 24px;
            font-weight: 700;
        }
        .active-box {
            background-color: #4CAF50;
            color: white;
        }
        .idle-box {
            background-color: #FFD700;
            color: black;
        }
        .spindle-box {
            background-color: #E0F7FA;
            color: black;
        }
        .alarm-box {
            background-color: #F44336;
            color: white;
        }
       /* Parameters styling */
        .parameters-container {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .parameters-box {
            flex: 1;
            padding: 8px;
            background-color: #ffffff;
            border-radius: 5px;
        }
        .parameters-box .title {
            font-size: 25px;
            color:#000000;
            font-weight: bold;
            text-align: center;
            margin-bottom: 5px;
        }
        .parameters-box .values {
            display: flex;
            justify-content: space-around;
            color: #000000;
            flex-wrap: wrap;
        }
        .parameters-box .parameter {
            color: #F44336;
            font-size: 16px;
            font-weight: 700;
            margin: 3px 5px;
        }
        /* Charts container */
        .charts-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .chart-box {
            flex: 1;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        /* Adjust padding for the whole app */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        /* Button styling */
        .stButton > button {
            background-color: #FFA500;
            color: white;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #FF8C00;
        }
    </style>
    """, unsafe_allow_html=True)
# Helper functions for outlier detection and correlation
def get_thresholds(data_, lower_percentile, upper_percentile):
    """Compute column-wise lower/upper threshold for outlier detection."""
    lower_thresh = np.percentile(data_, lower_percentile, axis=0)
    upper_thresh = np.percentile(data_, upper_percentile, axis=0)
    return lower_thresh, upper_thresh

def get_outlier_columns_for_row(data_, row_idx, lower_thresh, upper_thresh):
    """Return columns for row_idx that are < lower_thresh or > upper_thresh."""
    row_data = data_[row_idx, :]
    mask = (row_data < lower_thresh) | (row_data > upper_thresh)
    return np.where(mask)[0]

# def find_top_correlated_columns(data_, row_idx, col_idx, window_size=50, top_r=2):
#     """Find the top_r columns in a window correlated to `col_idx` column."""
#     nrows, ncols = data_.shape
#     start_idx = max(0, row_idx - window_size + 1)
#     subset = data_[start_idx : row_idx + 1, :]  # shape ~ (window_size, ncols)

#     outlier_data = subset[:, col_idx]
#     correlations = []
   
#     for other_col in range(ncols):
#         if other_col == col_idx:
#             # Skip correlation with itself
#             correlations.append((other_col, np.nan))
#             continue
   
#         other_data = subset[:, other_col]
#         if (np.std(outlier_data) == 0) or (np.std(other_data) == 0):
#             corr_val = 0.0
#         else:
#             corr_val = np.corrcoef(outlier_data, other_data)[0, 1]
#         correlations.append((other_col, corr_val))
   
#     correlations.sort(key=lambda x: abs(x[1]), reverse=True)
#     return correlations[:top_r]

def find_top_correlated_columns(data_, row_idx, col_idx, window_size=50, top_r=2):
    """Find the top_r columns in a window correlated to `col_idx` column."""
    nrows, ncols = data_.shape
    start_idx = max(0, row_idx - window_size + 1)
    subset = data_[start_idx : row_idx + 1, :]  # shape ~ (window_size, ncols)

    outlier_data = subset[:, col_idx]
    correlations = []
   
    for other_col in range(ncols):
        if other_col == col_idx:
            # Skip correlation with itself
            continue
   
        other_data = subset[:, other_col]
        if (np.std(outlier_data) == 0) or (np.std(other_data) == 0):
            continue  # Skip columns with zero standard deviation
        else:
            corr_val = np.corrcoef(outlier_data, other_data)[0, 1]
            # Only add non-NaN correlations
            if not np.isnan(corr_val):
                correlations.append((other_col, corr_val))
   
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top correlations or empty list if none found
    return correlations[:top_r]

def make_scatter_plotly(x_series, y_series, x_label, y_label, title):
    """Scatter plot with Plotly."""
    fig = px.scatter(
        x=x_series,
        y=y_series,
        title=title,
        labels={"x": x_label, "y": y_label}
    )
    return fig

def create_dashboard_html(current_time, machine_status, pred_probs, outlier_cols, next_status, steps_to_next_state,row_idx):
    """Generate the HTML for the dashboard."""
    # Extract prediction probabilities
    active_prob = f"{pred_probs[0]:.2f}"
    idle_prob = f"{pred_probs[1]:.2f}"
    spindle_prob = f"{pred_probs[2]:.2f}"
    alarm_prob = f"{pred_probs[3]:.2f}"

    # Map outlier column indices to parameter names
    outlier_cols_str = [param_mapping.get(col, f"Unknown ({col})") for col in outlier_cols]

    # If no outliers, add a placeholder
    if not outlier_cols_str:
        outlier_cols_str = ["None"]

    # For long period, use "NIL" for now
    long_period_params = ["NIL"]

    # Generate HTML content
    dashboard_html = f"""
    <div class="dashboard-header">
        <div class="logo">stryker®</div>
        <div class="title">Machine Dashboard with classes and Probabilities</div>
    </div>

    <div class="status-container">
        <div class="status-box">
            <div class="label">Time Stamp</div>
            <div class="value">{row_idx + 1}</div>
        </div>
        <div class="status-box">
            <div class="label">Current Machine Status</div>
            <div class="value active-status">{machine_status}</div>
        </div>
        <div class="status-box">
            <div class="label">Next Status Expected</div>
            <div class="value idle-status">{next_status}</div>
        </div>
        <div class="status-box">
            <div class="label">Next Status Expected In</div>
            <div class="value">{int(steps_to_next_state)}</div>
        </div>
    </div>

    <div class="probability-title">Machine state Probabilities</div>
    <div class="probability-container">
        <div class="probability-box active-box">
            <div class="label">Active</div>
            <div class="value">{active_prob}</div>
        </div>
        <div class="probability-box idle-box">
            <div class="label">Idle</div>
            <div class="value">{idle_prob}</div>
        </div>
        <div class="probability-box spindle-box">
            <div class="label">Spindle Warm Up</div>
            <div class="value">{spindle_prob}</div>
        </div>
        <div class="probability-box alarm-box">
            <div class="label">Alarm</div>
            <div class="value">{alarm_prob}</div>
        </div>
    </div>

    <div class="parameters-container">
        <div class="parameters-box">
            <div class="title">Current Out of range Parameters</div>
            <div class="values">
                {' , '.join(outlier_cols_str)}  <!-- Join the parameter names with a comma -->
            </div>
        </div>
        <div class="parameters-box">
            <div class="title">Long period Out of range Parameters</div>
            <div class="values">
                {' '.join([f'<span class="parameter">{param}</span>' for param in long_period_params])}
            </div>
        </div>
    </div>
    """
    return dashboard_html


def main():
    apply_custom_styling()

    with st.container():
        st.title("Machine Dashboard Demo")

    # if st.button("Show Dashboard"):
        # Initialize placeholder containers
        dashboard_container = st.empty()
        charts_container = st.empty()
       
        # Set up lower/upper thresholds for outlier detection
        lower_thresh, upper_thresh = get_thresholds(data, 5, 95)
       
        for row_idx in range(n_rows):
            # Get current time
            current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            
            # Get actual class and machine status
            actual_class = int(classes[row_idx])
            machine_status = label_mapping[actual_class]  # Map actual class to machine status

            # Get prediction probabilities
            pred_probs = predictions[row_idx]  # Predicted probabilities for the row
            
            # Get outlier columns (You should have implemented this function)
            outlier_cols = get_outlier_columns_for_row(data, row_idx, lower_thresh, upper_thresh)
            
            # Get next state and steps from state_transition_predictions.npy
            next_state = label_mapping.get(predicted_next_state[row_idx], "Unknown State")  # Safe access with default
            steps_to_next_state = predicted_steps[row_idx]  # Predicted steps for the current row

            # Create dashboard HTML (You should have implemented this function)
            dashboard_html = create_dashboard_html(
                current_time=current_time,
                machine_status=machine_status,
                pred_probs=pred_probs,
                outlier_cols=outlier_cols,
                next_status=next_state,
                steps_to_next_state=steps_to_next_state,
                row_idx=row_idx
            )
            
            # Display the dashboard (this is where you render the HTML)
            dashboard_container.markdown(dashboard_html, unsafe_allow_html=True)
                
            # Set up charts display
            with charts_container.container():
                # Create placeholder for charts
                st.markdown('<div class="charts-container">', unsafe_allow_html=True)
               
                col1, col2 = st.columns(2)
               
                # Create scatter plots if outliers exist
                if len(outlier_cols) > 0:
                    primary_col = outlier_cols[0]
                    top_corr = find_top_correlated_columns(data, row_idx, primary_col)
                   
                    start_idx = max(0, row_idx - 50 + 1)
                    subset = data[start_idx : row_idx + 1, :]
                    x_main = subset[:, primary_col]
                   
                    with col1:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        if len(top_corr) >= 1:
                            col1_idx, corr1 = top_corr[0]
                            if not np.isnan(corr1):
                                y_1 = subset[:, col1_idx]
                                fig1 = make_scatter_plotly(
                                    x_main, y_1,
                                    x_label=f"Column {primary_col}",
                                    y_label=f"Column {col1_idx}",
                                    title=f"Time Stamp: {row_idx}, Corr={corr1:.2f}"
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                   
                    with col2:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        if len(top_corr) >= 2:
                            col2_idx, corr2 = top_corr[1]
                            # No need to check for NaN since we filtered them out earlier
                            y_2 = subset[:, col2_idx]
                            fig2 = make_scatter_plotly(
                                x_main, y_2,
                                x_label=f"Column {primary_col}",
                                y_label=f"Column {col2_idx}",
                                title=f"Time Stamp: {row_idx}, Corr={corr2:.2f}"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.markdown('Not enough valid correlations found', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # If no outliers, display empty chart containers
                    with col1:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        st.markdown('No outliers detected in current row', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        st.markdown('No outliers detected in current row', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
               
                st.markdown('</div>', unsafe_allow_html=True)
               
            # Wait before updating to next row
            time.sleep(5)

            # Hide the title and button
            st.markdown("""
            <style>
            div[data-testid="stVerticalBlock"] > div:nth-child(1) {display: none;}
            div[data-testid="stButton"] {display: none;}
            [data-testid="stHeader"]{display: none;}
            </style>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
