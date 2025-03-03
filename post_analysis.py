"""
"""
#%%
## Import the libraries
import rasterio
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import seaborn as sns
from io import StringIO
import streamlit as st
import plotly.express as px
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import glob
import ee
import geemap
from tqdm import tqdm  # Import tqdm for the progress bar
from rasterio.plot import show
import geemap.colormaps as cm
import folium
from matplotlib.dates import DateFormatter, DayLocator
#from IPython.display import display
from branca.colormap import LinearColormap
import altair as alt
import geemap.foliumap as geema

#%%
#__________________________________________________POST ANALYSIS______________________________________________________________________________________
# Title of the Streamlit App
st.title("Paddy Flooding Detection using Sentinel 2 Analysis (2019-2024)")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
START_PLANTING = 46  # 15 Feb
END_PLANTING = 74    # 15 Mar
START_HARVESTING = 186  # 5 Jul
END_HARVESTING = 259    # 16 Sep

def read_github_csv(url):
    """Read CSV from GitHub URL."""
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return pd.read_csv(raw_url)

def process_rs_data(df):
    rs_df = df.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
    area_rs = rs_df.sum(axis=0)
    rs_df_combined = pd.DataFrame({
        'Time': area_rs.index,
        'Area(ha)': area_rs.values
    })
    rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
    rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
    rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
    rs_df_combined['DOY'] = rs_df_combined['Time'].dt.dayofyear
    return rs_df_combined

def plot_cumulative_area(df, title):
    """Plot cumulative flooded area."""
    years = df['Time'].dt.year.unique()
    fig, ax = plt.subplots(figsize=(10, 6))

    for year in years:
        year_df = df[df['Time'].dt.year == year]
        ax.plot(year_df['DOY'], year_df['Area(ha)'], marker='o', linestyle='-', label=f'RS Area {year}')

    ax.axvline(START_PLANTING, color='blue', linestyle='--', label='Start Planting (15 Feb)')
    ax.axvline(END_PLANTING, color='green', linestyle='--', label='End Planting (15 Mar)')
    ax.axvline(START_HARVESTING, color='orange', linestyle='--', label='Start Harvesting (5 Jul)')
    ax.axvline(END_HARVESTING, color='red', linestyle='--', label='End Harvesting (16 Sep)')

    ax.set_title(title)
    ax.set_xlabel('Day of Year (DOY)')
    ax.set_ylabel('Area (ha)')
    ax.grid(True)
    ax.legend()
    
    return fig

def process_data(urls, title):
    """Process and plot data from a list of URLs."""
    dataframes = []
    for url in urls:
        try:
            df = read_github_csv(url).drop(columns=['flooding_date'], errors='ignore')
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error reading {url}: {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        processed_df = process_rs_data(combined_df)
        
        fig = plot_cumulative_area(processed_df, title)
        
        col1, col2 = st.columns([3, 1])
        col1.pyplot(fig)
        col2.subheader("Data Sample")
        col2.write(processed_df.head(20))
    else:
        st.warning(f"No {title} data available.")

# Dagana Flooding Data
st.header("**1. Dagana Plots**")
dagana_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/flooding_data_{year}.csv' for year in range(2019, 2025)]
process_data(dagana_urls, "2019-2024 Cumulative Flooded Areas using Dagana Plots")

# agCelerant Data
st.header("**2. agCelerant Plots**")
agcelerant_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/combined_flooding_data_{year}.csv' for year in range(2019, 2025)]
process_data(agcelerant_urls, "2019-2024 Cumulative Flooded Areas using agCelerant Plots")


#___________________________________________________________credit information________________________-


# Header for Credit Information
st.header("3. Credit Information")

# Initial Hypotheses
st.subheader("Initial Hypotheses")
st.markdown("""
- **agCelerant Data**:
    - Increase in flooding → Early promise of credit.
    - Decrease in flooding → Delay in the promise of credit.
""")

# Learnings from Data Exploration
st.subheader("Learnings from Exploration of the Data")

st.subheader("First Spike Hypothesis")
st.markdown("""
- Some GIEs might have internal financial reserves that are used.
- Some GIEs have good creditworthiness and could use this to get a loan from the bank.
""")

st.subheader("Second Spike Hypothesis")
st.markdown("""
- Reaction by the second category of farmers (those who lack internal reserves and thin file GIEs) to the promise of credit.
- Or the same group has learned about the decisions of the banks on credit issuance.
""")

st.subheader("Early Flooding Hypothesis")
st.markdown("""
- Early flooding explains/reflects the credit disbursement process.
- Everyone starts the cropping cycle without any payment. The cycle is based on promises or commitments by the GIE to get credit.
- Some unions have reserves to start planting before funds are released.
- The early start does not necessarily mean the money has been released yet.
""")

# New Hypothesis
st.subheader("New Hypothesis")
st.markdown("""
- **Creditworthiness and Reserves of GIE Explain the Monotonic Pattern of Flooding**
    - Does the absence of a plateau imply reserves?
    - Is there a significant difference between the lengths of the plateaus? Does a significant difference in the length of the plateau reflect different access to credit? 
    - Extract the date of the credit committee for the Dry hot seasons from 2017-2024 years.
    - Explore the topology of GIE.
""")

# Action Points
st.subheader("Action Points")
st.markdown("""
- Consider **quantitative data analysis**: This approach is more process-based and accumulates evidence from many points to understand how the accumulation of events (like floods) reflects credit disbursement timelines.
- Zoom in to analyze **GIEs**:
    - See their performance in terms of their decision to go into production.
    - The commitment of credit & the farmers' response.
- Extract boundaries and calculate the total areas for confirmation with "Soule".
- Investigate the date of credit committee meetings and decisions for each year.
- Investigate when money starts flowing for them (GIE).
""")

st.markdown("General Information: The dataset contains 4,608 rows and 258 columns. "
            "The `parcel_cod` column has 596 unique values, indicating multiple parcels. "
            "The `operating_account` column contains 163 unique types of operations, "
            "with 'Assurance CNAAS' being the most frequent. "
            "There are missing data in `credit_auth_date` and `credit_exec_date`.")



#_____________________________________________________________uuuuuuuuuuuuuu____________________________
file_url ='https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/merged_cr_rs_data.csv'
#read_github_csv(url)
combined_df_credit = read_github_csv(file_url)#pd.read_csv(file, index_col=0,na_values=np.nan)

st.markdown("**Sample of the Credit Data:**")
st.dataframe(combined_df_credit.head(20))

#_____________________________________GIE analysis________________________________________-_______________________________
st.header("4. GIE Analysis")
# Load the merged data
data = combined_df_credit.copy()
# Convert None to empty strings
data = data.fillna('')

# Filter out rows with empty strings in the specified columns
has_credit_details = data[
    (data['credit_req_date'] != '') &
    (data['credit_auth_date'] != '') &
    (data['credit_exec_date'] != '')
]
#data = data.replace({None: np.nan})
# Move filters to the sidebar
#st.sidebar.header("GIE Based Analysis-Has")
# Filter GIEs with all credit details (credit_req_date, credit_auth_date, credit_exec_date)
#has_credit_details = data.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date'],how='all')
gies_with_credit_details = data['gie_name'].unique()
#%%
# Display number of all unique GIEs
all_gies = data['gie_name'].unique()
st.write(f"- Total number of GIEs available for analysis: {len(all_gies)}")
# Filter GIEs with all credit details (credit_req_date, credit_auth_date, credit_exec_date)
#has_credit_details = data.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date'])
gies_with_credit_details = has_credit_details['gie_name'].unique()
# Display number of GIEs with complete credit details
st.write(f"- Number of GIEs with complete credit details: {len(gies_with_credit_details)}")
st.write('-Time to execution: the number of days from credit request to execution')
# Select GIE for analysis in the sidebar
selected_gie = st.selectbox('Select GIE:', has_credit_details['gie_name'].unique())
st.markdown(f"**Details for GIE: {selected_gie}**")
filtered_data = has_credit_details[has_credit_details['gie_name'] == selected_gie]

# Convert the 'credit_req_date', 'credit_auth_date', and 'credit_exec_date' columns to datetime for analysis
filtered_data['credit_req_date'] = pd.to_datetime(data['credit_req_date'], errors='coerce')
filtered_data['credit_auth_date'] = pd.to_datetime(data['credit_auth_date'], errors='coerce')
filtered_data['credit_exec_date'] = pd.to_datetime(data['credit_exec_date'], errors='coerce')

# Calculate the time between request and execution
filtered_data['time_to_execution'] = (filtered_data['credit_exec_date'] - filtered_data['credit_req_date']).dt.days

# Display basic statistics for the time to execution
time_to_execution_stats = filtered_data['time_to_execution'].describe()

# Plotting a histogram of time to execution
fig2 = plt.figure(figsize=(10, 6))
plt.hist(filtered_data['time_to_execution'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title(f'Distribution of Time to Execution for Credit Requests for GIE: {selected_gie}')
plt.xlabel('Days from Credit Request to Execution')
plt.ylabel('Frequency')
plt.grid(True)

# Show the plot
plt.show()
st.pyplot(fig2)

# Process RS data (you can define this function)
gie_rs_df = process_rs_data(filtered_data)

# Plotting the Cumulative Area for the selected GIE
years = gie_rs_df['Time'].dt.year.unique()
fig, ax = plt.subplots(figsize=(10, 6))

for year in years:
    year_df = gie_rs_df[gie_rs_df['Time'].dt.year == year]
    ax.plot(year_df['DOY'], year_df['Area(ha)'], marker='o', linestyle='-', label=f'RS Area {year}')

start_planting = datetime.strptime('2023-02-15', '%Y-%m-%d').timetuple().tm_yday
end_planting = datetime.strptime('2023-03-15', '%Y-%m-%d').timetuple().tm_yday
start_harvesting = datetime.strptime('2023-07-05', '%Y-%m-%d').timetuple().tm_yday
end_harvesting = datetime.strptime('2023-09-16', '%Y-%m-%d').timetuple().tm_yday

ax.axvline(start_planting, color='blue', linestyle='--', label='Start Planting (15 Feb)')
ax.axvline(end_planting, color='green', linestyle='--', label='End Planting (15 Mar)')
ax.axvline(start_harvesting, color='orange', linestyle='--', label='Start Harvesting (5 Jul)')
ax.axvline(end_harvesting, color='red', linestyle='--', label='End Harvesting (16 Sep)')

ax.set_title(f'2019-2024 Cumulative Flooded Areas for GIE: {selected_gie}')
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylabel('Area (ha)')
ax.grid(True)
ax.legend()
st.pyplot(fig)


# Bar chart for Irrigation Type
st.subheader(f"Count of Irrigation Types for: {selected_gie}")
irrigation_type_counts = filtered_data['irrigation_type'].value_counts()
fig_irrigation, ax_irrigation = plt.subplots(figsize=(8, 4))
ax_irrigation.bar(irrigation_type_counts.index, irrigation_type_counts.values, color='skyblue')
ax_irrigation.set_title('Irrigation Type Count')
ax_irrigation.set_xlabel('Irrigation Type')
ax_irrigation.set_ylabel('Count')
ax_irrigation.grid(True)
st.pyplot(fig_irrigation)

st.title(f'Operations accounts for {selected_gie}')
st.write(' Visualizes the credit request, authorization, and execution process for different operating accounts ( eg gasoil irrigation) in a timeline format. This allows you to easily track the progress of each operating account’s credit process over time.')
# Prepare the data for timeline visualization
timeline_data = []
for idx, row in filtered_data.iterrows():
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_req_date'], Finish=row['credit_auth_date'], Stage='Credit Requested'))
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_auth_date'], Finish=row['credit_exec_date'], Stage='Credit Authorized'))
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_exec_date'], Finish=row['credit_exec_date'], Stage='Credit Executed'))

# Convert the timeline data into a DataFrame
timeline_df = pd.DataFrame(timeline_data)

# Custom color map for stages
custom_colors = {
    'Credit Requested': 'dodgerblue',
    'Credit Authorized': 'orange',
    'Credit Executed': 'green'
}

# Create a Gantt-style timeline using Plotly with customizations
fig_timeline = px.timeline(
    timeline_df,
    x_start="Start",
    x_end="Finish",
    y="Task",
    color="Stage",
    color_discrete_map=custom_colors,  # Use custom colors for each stage
    title=f"Credit Request, Authorization, and Execution Timeline for GIE: {selected_gie}",
    hover_data={
        'Start': '|%B %d, %Y',  # Customize hover date format
        'Finish': '|%B %d, %Y',  # Customize hover date format
        'Stage': True,
        'Task': True
    },
    labels={"Task": "Operation Account", "Stage": "Credit Stage"}
)

# Customizing the layout
fig_timeline.update_layout(
    xaxis_title="Date",
    yaxis_title="Operation Account",
    showlegend=True,
    height=600,
    margin=dict(l=100, r=50, t=50, b=50),  # Adjust margins for readability
    font=dict(family="Arial, sans-serif", size=12)  # Customize font
)

# Improve x-axis readability
fig_timeline.update_xaxes(
    tickformat="%b %d, %Y",  # Date format on the x-axis
    tickangle=45,  # Rotate x-axis labels
    nticks=20  # Control number of ticks
)

# Display the timeline
st.plotly_chart(fig_timeline)

# Assuming 'melted_data' is your DataFrame and it has a 'geometry' column
# Load GeoDataFrame (GIE credit and RS data with geometries)
st.dataframe(filtered_data)

#________________________________________________________PPPP___________________________________
# Prediction
st.header('4. Estimation of the Flooded area using growth Models')
#st.subheader('**Logistics groth model**')
# Title and description
st.title("Growth Curve: Logistic")

# Equation
st.markdown("### Equation:")
st.latex(r"P(t) = \frac{K}{1 + \left( \frac{K - P_0}{P_0} \right) e^{-rt}}")

# Explanation of the terms
st.markdown("""
Where:
- **P(t)**: Population or size at time **t**,
- **K**: Carrying capacity (the maximum size the population can reach),
- **P₀**: Initial population or size,
- **r**: Growth rate,
- **t**: Time.
""")

# Shape explanation
st.markdown("""
### Shape:
The logistic curve is S-shaped (sigmoidal). It starts with an exponential growth phase, 
slows down as the population approaches the carrying capacity, and eventually levels off at the carrying capacity.
""")

# Growth dynamics
st.markdown("""
### Growth Dynamics:
- **Initial Phase**: Rapid growth (exponential).
- **Middle Phase**: Growth rate decreases as resources become limited.
- **Final Phase**: Growth slows down and asymptotically approaches the carrying capacity, **K**.
""")

#plot the prediction and the data
def process_rs_data(dff):
    rs_df = dff.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))  # Date columns
    area_rs = rs_df.sum(axis=0)  # Summing values row-wise
    rs_df_combined = pd.DataFrame()
    rs_df_combined['Time'] = list(area_rs.index)
    rs_df_combined['Area(ha)'] = list(area_rs.values)
    rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
    rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
    rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
    rs_df_combined['DOY'] = rs_df_combined['Time'].apply(lambda x: x.timetuple().tm_yday)
    return rs_df_combined

    # Process both datasets
dag_rs_df = process_rs_data(read_github_csv('https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/flooding_data_2024.csv'))
file_log ='https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/prediction_growth_ts_first_attempt.csv'
data_log = read_github_csv(file_log)
data_log['Time'] = pd.to_datetime(data_log['time_t'])
data_log['Area(ha)'] = data_log['area_t']
#data_log['Class'] = 'log_2024'
#data_log['Time'] = pd.to_datetime(data_log['Time'])
data_log['DOY'] = data_log['Time'].apply(lambda x: x.timetuple().tm_yday)

fig, ax = plt.subplots(figsize=(10, 6))

combined_df_g = pd.concat([dag_rs_df,data_log],axis = 0)
# Plot each class in the 'Class' column
for cls in combined_df_g['Class'].unique():
    class_data = combined_df_g[combined_df_g['Class'] == cls]
    ax.plot(class_data['DOY'], class_data['Area(ha)'], marker='x', linestyle='--', label=cls)

# Customize plot
ax.set_title(f'Cumulative Flooded Areas {year}', fontsize=10)
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylabel('Cumulative Flooded Area (ha)')
ax.legend(title='Classes')
ax.grid(True)
plt.show()
st.pyplot(fig)

