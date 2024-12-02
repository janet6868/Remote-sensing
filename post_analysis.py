

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
from IPython.display import display
from branca.colormap import LinearColormap
import altair as alt
import geemap.foliumap as geema
from geemap.basemaps import GoogleMapsTileProvider


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
# Title of the Streamlit App
st.title("Paddy Flooding Detection using Sentinel 2 Analysis (2019-2024)")
#__________________________________________________FLOODING DETECTION______________________________________________________________________________________
#@title Running workflow 2022

# Authenticate and initialize the Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-janet')
# Define the grid and region of interest
grid = ee.FeatureCollection("projects/ee-janet/assets/senegal/52_grid_dagana")
init_dagana = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana")

# Additional layers
dagana_reservoir = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_reservoir")
dagana_water = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_water")
dagana_riverbanks = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_riverbanks")
dagana_wetland = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_wetland")
exclusion_area = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_exclusion_region")

exclusion_areas = dagana_riverbanks.geometry() \
    .union(dagana_wetland.geometry()) \
    .union(dagana_reservoir.geometry()) \
    .union(dagana_water.geometry())

# Subtract exclusion areas from the initial Dagana region
dagana = init_dagana.geometry().difference(exclusion_areas)#.difference(exclusion_area.geometry())

# Get the bounding box and center of the ROI for the Folium map
roi_bounds = dagana.bounds().getInfo()['coordinates'][0]
center_lat = (roi_bounds[0][1] + roi_bounds[2][1]) / 2
center_lon = (roi_bounds[0][0] + roi_bounds[2][0]) / 2

# Create a map
m = geemap.Map(center=[center_lat, center_lon], zoom=10)
# Add basemap
m.add_basemap('Esri.WorldImagery')

def run_detection_flooding(aoi, grid, start_date, end_date, year):
    # Function to calculate MNDWI for Sentinel-2
    def calculate_mndwi_s2(image):
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        return image.addBands(mndwi)

    # Function to calculate NDWI for Sentinel-2
    def calculate_ndwi_s2(image):
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndwi)

    # Function to calculate AWEI for Sentinel-2
    def calculate_awei_s2(image):
        awei = image.expression(
            '4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)',
            {
                'GREEN': image.select('B3'),
                'SWIR1': image.select('B11'),
                'NIR': image.select('B8'),
                'SWIR2': image.select('B12')
            }
        ).rename('AWEI')
        return image.addBands(awei)

    # Function to add s2cloudless cloud probability to Sentinel-2 imagery
    def add_cloud_probability(image):
        # Load the s2cloudless cloud probability image for the same time as the Sentinel-2 image
        cloud_probability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterBounds(image.geometry()) \
            .filterDate(image.date(), image.date().advance(1, 'day')) \
            .first()  # Get the first image in the filtered collection

        # Add the cloud probability as a band to the original image
        return image.addBands(cloud_probability.rename('cloud_prob'))

    # Function to mask clouds using s2cloudless cloud probability
    def mask_clouds_s2cloudless(image, cloud_prob_threshold=30):
        # Add cloud probability to the image
        image = add_cloud_probability(image)

        # Create a cloud mask where cloud probability is below the threshold
        cloud_mask = image.select('cloud_prob').lt(cloud_prob_threshold)

        # Apply the cloud mask to the image
        return image.updateMask(cloud_mask)

    # Function to mask clouds in Sentinel-2 imagery
    def mask_clouds_s2(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)

    # Function to process dates every 5 days
    def enhanced_date_processing(start_date, end_date, interval_days=5):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = []
        while start_date <= end_date:
            date_list.append(start_date.strftime("%Y-%m-%d"))
            start_date += timedelta(days=interval_days)
        return date_list
   
    # Function to calculate flood area for each grid cell
    def calculate_grid_flood_area(flood_mask, grid,date):
        def calculate_area(feature):
            area = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=10,
                maxPixels=1e13
            )
            area_ha = area.getNumber('constant').divide(10000).format('%.2f')
            #return feature.set('flood_area_ha', area_ha)
            return feature.set({'flood_area_ha': area_ha, 'date': date})
        return grid.map(calculate_area)

    # Function to get day of year
    def get_doy(date_string):
        date = datetime.strptime(date_string, '%Y-%m-%d')
        return date.timetuple().tm_yday

    def extract_flood_data(features, date):
        flood_data = []
        for feature in features:
            grid_id = feature['properties']['ID']
            flood_area_ha = feature['properties'].get('flood_area_ha', 0)
            flood_data.append({
                'date': date,
                'grid_id': grid_id,
                'flood_area_ha': flood_area_ha,
                **{k: v for k, v in feature['properties'].items() if k != 'flood_area_ha'}
            })
        return flood_data

    def create_flood_dataframe(flood_data):
        df = pd.DataFrame(flood_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date', 'grid_id'], inplace=True)
        return df

    def process_flood_results(df, grid_properties):
        df['flood_area_ha'] = pd.to_numeric(df['flood_area_ha'], errors='coerce')
        df['flood_area_ha'].fillna(0, inplace=True)
        if isinstance(grid_properties, pd.DataFrame):
            final_df = pd.merge(df.reset_index(), grid_properties, on='grid_id', how='left')
        else:
            final_df = df
        return final_df


    dataset = ee.Image('JRC/GSW1_4/MonthlyHistory/2021_01').clip(dagana)
    # Select water and create a binary mask
    water = dataset.select('water').eq(2)
    masked = water.updateMask(water)
    date_ranges = enhanced_date_processing(start_date, end_date)
    #   .map(mask_clouds_s2) \ .map(mask_clouds_s2cloudless)\
    def process_each_date(aoi, date):
        start_period = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=5)
        end_period = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=5)
        s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(aoi) \
                    .filterDate(start_period, end_period) \
                    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 18)) \
                    .map(mask_clouds_s2) \
                    .map(calculate_mndwi_s2)\
                   # .map(calculate_awei_s2)
        # Check if the image collection is empty
        if s2_sr_col.size().getInfo() == 0:

            print(f"No images found for date {date} within the specified cloud coverage.")
            return None
        # Mosaic the images
        mosaic = s2_sr_col.mosaic().clip(dagana)
        mosaic_ = mosaic.updateMask(water.Not())
        #Threshold each index
        mndwi_mask = mosaic_.select('MNDWI').gt(0)#updateMask(water_areas.Not()
        return mndwi_mask


    # Add this new function for exporting images
    def export_image(image, description, region, folder):
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            scale=10,
            region=region,
            maxPixels=1e13
        )
        task.start()
        print(f"Started export task for {description}")
    first_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(dagana).first()
    projection = first_image.select('B2').projection()
    # Function to process and visualize flooding, mapping only at the end of each month
    def process_and_visualize_flooding(aoi, date_ranges, grid):
        flood_data = []
        cumulative_flood_mask = ee.Image(0).reproject(crs=projection, scale=10).clip(aoi)
        flood_vis_params = {
            'min': min([get_doy(d) for d in date_ranges]),
            'max': max([get_doy(d) for d in date_ranges]),
            'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }

        current_month = None

        # Initialize the progress bar
        for i, date in tqdm(enumerate(date_ranges), total=len(date_ranges), desc="Processing Dates"):
            current_mndwi = process_each_date(aoi, date)
            if current_mndwi is not None:
                doy = get_doy(date)
                base_date_mask = current_mndwi
                #mask1 = base_date_mask.And(cumulative_flood_mask.eq(1))
                cumulative_flood_mask = cumulative_flood_mask.where(base_date_mask.And(cumulative_flood_mask.eq(0)), doy)
                # Calculate the flood area for the grid at each 5-day interval
                grid_with_flood_area = calculate_grid_flood_area(cumulative_flood_mask.gt(0), grid, date) #.updateMask(cumulative_flood_mask.gt(0))
                flood_data.extend(extract_flood_data(grid_with_flood_area.getInfo()['features'], date))
                # Get the month of the current date
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if current_month is None:
                    current_month = date_obj.month
                # Check if the next date is in a new month or if it's the last date
                is_end_of_month = (i == len(date_ranges) - 1) or (datetime.strptime(date_ranges[i + 1], "%Y-%m-%d").month != current_month)
                if is_end_of_month:
                    # Add layer to map
                    # m.add_layer(cumulative_flood_mask.updateMask(cumulative_flood_mask).gt(0),
                    #             flood_vis_params, f'Flooding Progression up to {date}')
                    m.add_layer(cumulative_flood_mask.updateMask(cumulative_flood_mask.gt(0)),
                                  flood_vis_params, f'Flooding Progression up to {date}')

                    export_folder = "fis_flooding_maps"
                    # Export the cumulative flood mask
                    export_image(
                        cumulative_flood_mask.updateMask(cumulative_flood_mask.gt(0)),
                        f'Flooding_map_{date}',
                        aoi,
                        export_folder
                   )
                    # Reset current_month for the next month
                    if i != len(date_ranges) - 1:
                        current_month = datetime.strptime(date_ranges[i + 1], "%Y-%m-%d").month
            else:
                print(f"Skipping date {date} due to no valid MNDWI")
        m.add_colorbar(flood_vis_params, label="Day of the year",
                       orientation="horizontal",
                       layer_name="Flooding detection")
        return flood_data
    
    flood_data = process_and_visualize_flooding(dagana, date_ranges, grid)
    df = create_flood_dataframe(flood_data)
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Pivot the table
    df_pivoted = df.pivot(index='grid_id', columns='date', values='flood_area_ha')
    columns_date = df_pivoted.columns
    maxValueIndex1 = df_pivoted[columns_date].idxmax(axis=1)
    df_pivoted['flooding_date'] = maxValueIndex1

    # Assuming the columns_to_keep exist in your data
    columns_to_keep = ['ID', 'LatNP', 'Latitude', 'LonNP', 'Longitude', 'nasapid']
    df_other = df.drop_duplicates(subset=['grid_id'])[columns_to_keep + ['grid_id']].set_index('grid_id')
    df_pivoted = df_pivoted.rename_axis(index='grid_id')
    df_final = df_other.join(df_pivoted)
    df_final = df_final.reset_index()
    date_columns = [col for col in df_final.columns if col not in columns_to_keep + ['grid_id', 'index']]
    df_final = df_final[columns_to_keep + ['grid_id'] + sorted(date_columns)]

    output_file_name = f'floodingData_{year}.csv'
    df_final.to_csv(output_file_name, index=False)
    # Display the map
    display(m)
    # Add additional layers to the map
    m.addLayer(dagana_reservoir, {'color': 'blue'}, 'Dagana Reservoir')
    m.addLayer(dagana_water, {'color': 'cyan'}, 'Dagana Water')
    m.addLayer(dagana_riverbanks, {'color': 'green'}, 'Dagana Riverbanks')
    m.addLayer(dagana_wetland, {'color': 'brown'}, 'Dagana Wetland')


year = '2024'
# Define the date range for processing
start_date = '2024-01-17'
end_date = '2024-02-28'
# Process flooding data and create DataFrame for analysis
run_detection_flooding(aoi= dagana, grid=grid, start_date=start_date, end_date=end_date, year=year)
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

