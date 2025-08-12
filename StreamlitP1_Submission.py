###Library Imports 
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import joblib
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import math
import os, requests
from huggingface_hub import hf_hub_download
import json
from openai import OpenAI
###_________________________________________________________________________________
def _get_openai_client():
    api_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if not api_key or not isinstance(api_key, str) or not api_key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY missing or invalid in secrets/environment.")
    return OpenAI(api_key=api_key)
REPO_ID = "ZednemXela/df_2024"  
FILENAME = "df_2024.csv"                       
REPO_TYPE = "dataset"
###Caches the dataset for faster load times. 
@st.cache_data(show_spinner="Downloading dataset from Hugging Face…")
def load_df():
    token = st.secrets.get("HF_TOKEN", None) 
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type=REPO_TYPE
    )
    ### Read using the downloaded path (NOT just "df_2024.csv")
    try:
        df = pd.read_parquet(local_path)
    except Exception:
        df = pd.read_csv(local_path, low_memory=False)
    return df, local_path

df_2024, cache_path = load_df()
### For dev, gives information on where the large dataset is loaded from. For submission this is commented out. 
### st.write("Loaded from:", cache_path) 
###___________________________________________________________________________________
### Open AI Helper Function
def _openai_client():
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")
    return OpenAI(api_key=key)

def ai_summary_with_openai(neighborhood, wx, picks, shortlist, model="gpt-3.5-turbo"):
    client = _get_openai_client()
    resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    )
    system = (
        "You are a concise food guide. Use the weather and the shortlist of nearby restaurants "
        "to make 2–4 specific cuisine suggestions and a friendly rationale. "
        "Keep it under 120 words. No markdown bullets—use short sentences."
    )
    user = (
        f"Neighborhood: {neighborhood}\n"
        f"Weather: {wx.get('description','?')}, {wx.get('temp_f','?')}°F, "
        f"precip {wx.get('precip_in','?')} in, wind {wx.get('wind_mph','?')} mph.\n"
        f"Suggested cuisines: {', '.join(picks) if picks else 'Any'}\n\n"
        f"Nearby restaurants (name – cuisine – distance mi):\n{shortlist}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.4,
        max_tokens=240,
    )
    return resp.choices[0].message.content.strip()


### Streamlit Heading#####
st.set_page_config(page_title="CIS9660 - Final Projects", layout="wide")
###________________________________________________________________________________
### Import necessary datasets
### Regression - Project 1
df = pd.read_csv("Streamlit_dataset.csv")
###Project 2 - Classification  
df_2024['sale_date'] = pd.to_datetime(df_2024['sale_date'], errors='coerce')
d24 = df_2024[df_2024['sale_date'].dt.year == 2024].copy()
###______________________________________________________________________________
### Project 2 - AI Agent Function 

### initiates helper function 1, geocode place. This turns neighborhoods into lat & long coordinates. 
def geocode_place(name: str):
    ### Nominatin function, part of openstreetsmap geocoding services. 
    geolocator = Nominatim(user_agent="streamlit_weather_food")
    ### Uses .geocode(name) to find coordinates for the place. 
    loc = geolocator.geocode(name)
    ### If no location is found an error is raised. 
    if not loc:
        raise ValueError(f"Could not geocode: {name}")
        ### returns neighborhood as lat and long functions. 
    return float(loc.latitude), float(loc.longitude)
### Initiates helper function 2, grabs the current weather at specific lat and long. 
def get_weather(lat: float, lon: float):
     ### Sets an API end point 
    url = "https://api.open-meteo.com/v1/forecast"
    ### Creates a dictonary with the following keys, using standard conventions.  
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
        "temperature_unit": "fahrenheit",   # °F
        "windspeed_unit": "mph",            # mph
        "precipitation_unit": "inch",       # inches
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    cur = r.json().get("current", {})

    ### Creates a different dictonary for the weather. Giving the following attributes that are below. 
    code_map = {
        0:"Clear", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
        61:"Light rain", 63:"Rain", 65:"Heavy rain",
        71:"Light snow", 73:"Snow", 75:"Heavy snow",
        80:"Rain showers", 81:"Heavy rain showers", 82:"Violent rain showers",
        95:"Thunderstorm"
    }
    code = cur.get("weather_code", 0)
    ### Returns the results in dictionary format. 
    return {
        "temp_f": cur.get("temperature_2m"),
        "precip_in": cur.get("precipitation", 0),
        "wind_mph": cur.get("wind_speed_10m", 0),
        "description": code_map.get(code, f"Code {code}"),
        
    }


### This function finds restaurants using OpenStreetMap's API and returns the results as a pandas dataframe. 
def overpass_restaurants(lat: float, lon: float, radius_miles: float = 1.0, limit: int = 80) -> pd.DataFrame:
    """
    Find nearby restaurants within radius_miles (miles) using OpenStreetMap (Overpass).
    Returns a DataFrame with name, cuisine, distance_miles, and OSM link.
    """
    ### The defualt setting is in metric so it is then converted to  miles.
    radius_m = radius_miles * 1609.34

    ### Creates a query using the json data and provides a search radius. 
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
      way["amenity"="restaurant"](around:{radius_m},{lat},{lon});
      relation["amenity"="restaurant"](around:{radius_m},{lat},{lon});
    );
    out center tags {limit};
    """
    ### Sends the query
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=40)
    ### Raises errors if the request fails
    r.raise_for_status()
    ### Parses JSON response. 
    js = r.json()
    ### extracts name, cuisine, and addresses and links them to the OSM page. If a restaurant doesn't have a name it is skipped. 
    rows = []
    for el in js.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        # nodes: lat/lon; ways/relations: center.lat/center.lon
        lat_el = el.get("lat") or (el.get("center") or {}).get("lat")
        lon_el = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat_el is None or lon_el is None:
            continue

        cuisine = tags.get("cuisine", "")
        addr = tags.get("addr:street", "")
        osm_url = f"https://www.openstreetmap.org/{el.get('type','node')}/{el.get('id')}"

        rows.append({
            "name": name,
            "cuisine": cuisine,
            "address": addr,
            "lat": lat_el,
            "lon": lon_el,
            "osm_url": osm_url
        })
    ### Creates a dataframe and calculates the distance (in miles)
    df = pd.DataFrame(rows)
    if not df.empty:
        ### distance in miles
        df["distance_miles"] = df.apply(
            lambda r: round(geodesic((lat, lon), (r["lat"], r["lon"])).miles, 2),
            axis=1
        )
        df = df.sort_values("distance_miles").reset_index(drop=True)

    return df
### Creates a new function that asks for neighborhood, weather, picks, and short list text. 
def ai_summary_with_openrouter(neighborhood: str, weather: dict, picks: list, shortlist_text: str,
                               model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> str:
    """
    Calls OpenRouter chat completions and returns a concise, friendly summary.
    Requires OPENROUTER_API_KEY in st.secrets or environment.
    """
    ### Get API key safely from Streamlit secrets (or env if running locally)
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to Streamlit Secrets.")
    ### Sets request headers
    base_url = "https://openrouter.ai/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        ### Optional but nice UX:
        "HTTP-Referer": "https://streamlit.app",  
        "X-Title": "Neighborhood Weather Food Guide",
    }
    ### Builds API Prompt
    system_msg = (
        "You are a helpful local food guide. Be accurate and concise. "
        "Do not invent restaurants beyond the provided list."
    )
    ### This will be the default user prompt. 
    user_prompt = f"""
Neighborhood: {neighborhood}
Weather: {weather.get('description')} — {weather.get('temp_f')}°F, precip {weather.get('precip_in')} in, wind {weather.get('wind_mph')} mph.
Weather-aware cuisines to consider: {", ".join(picks) if picks else "any"}.

Nearby restaurants (name — cuisine (distance mi)):
{shortlist_text}

Write a 3–7 sentence recommendation explaining how the weather influences the vibe and which cuisines fit.
Call out 2–3 specific restaurants from the list as examples. Keep it positive and practical.
"""
    ### API payload information 
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 350,
        "temperature": 0.7,
    }
    ### Call OpenRouter API. 
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


###______________________________________________________________
# Sidebar navigation
st.set_page_config(page_title="CIS9660 - Projects", layout="wide")

### Sidebar navigation
section = st.sidebar.radio("Navigation", [
    "Introduction - Project 1 Regression",
    "Exploratory Analysis and Cleaning",
    "Regression - Total Price",
    "Regression - Total Units",
    "Conclusion",
    "Introduction - Project 2 Classification",
    "Multi-Classification Models",
    "Model Results",
    "AI Agent",
    
])

### Global Variables_____________________________________________________________
### Total Price Metrics - Project 1 Regression
r2_raw = 0.0927
r2_log = 0.0049
r2_log10 = 0.0048
r2_df = pd.DataFrame({
    "Target Transformation": ["Raw Price", "Log(Price)", "Log₁₀(Price)"],
    "R² Score": [r2_raw, r2_log, r2_log10]
})

###_____________________________________________________________________________
###Prep for visuals for project 2

### Bins the data by price categories.
### Defines bins 
labels = ['Low', 'Medium', 'High', 'Very High']
###Checks to see if the price_category exists within the dataframe. 
if 'price_category' not in df_2024.columns:
  ### Defines a quartile function. Local variable s = sales prices for a single borough. 
  def quartiles_by_rank_safe(s):
      ### handles any missing vlaues
      out = pd.Series(pd.NA, index=s.index, dtype="object")
      m = s.notna(); n = int(m.sum())
      if n == 0: return out
      ### Assigns ranks
      r = s.rank(method='first')
      ### Handlee single unique value cases.  
      if s[m].nunique() == 1:
          out[m] = 'Medium'
          return out
      ### Cuts into quartiles
      q = min(4, n)
      out[m] = pd.qcut(r[m], q, labels=labels[:q], duplicates='drop').astype(str)
      return out
  ### Applies the function per borough creating 4 bins for each borough, each bin holds 25% of the data. 
  df_2024['price_category'] = (
      df_2024.groupby('borough', group_keys=False)['sale_price']
              .apply(quartiles_by_rank_safe)
  )
  ### Stores data logically Low < Medium < High < Very High
  df_2024['price_category'] = pd.Categorical(df_2024['price_category'],
                                              categories=labels, ordered=True)

###_____________________________________________________________________________
###Total Unit Regression model import 
##Imports of regression model from analysis workbook
model = joblib.load('multiple_linear_model.pkl')
scaler = joblib.load('scaler.pkl')
###Creating variables for visuals within Streamlit workbook
features = [
    'gross_square_feet', 'land_square_feet', 'year_built', 'sale_price',
    'units_per_sqft_pct', 'net_square_feet',
    'borough_Brooklyn', 'borough_Manhattan', 'borough_Queens', 'borough_Staten Island',
    'borough_The Bronx', 'floor_area_ratio'
]

X_all = df[features]
y_all = df["total_units"]
X_all_scaled = scaler.transform(X_all)
y_pred_all = model.predict(X_all_scaled)



###_____________________________________________________________________________
### Total Unit Metrics

### Creating dictionaries for metrics of training veresus testing data. 
metrics_data_units = {
    "Metric": ["Mean Squared Error (MSE)", 
               "Root Mean Squared Error (RMSE)", 
               "Mean Absolute Error (MAE)", 
               "R-squared (R²) Score"],
    "Training": [232.5162, 15.2485, 1.6095, 0.9260],
    "Testing": [14.7853, 3.8452, 1.4517, 0.9611]
}
### Creates a dataframe out of the dictionaries
metrics_df = pd.DataFrame(metrics_data_units)


###_____________________________________________________________________________

### Section edit
if section == "Introduction - Project 1 Regression":
    # Formatted using HTML
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Introduction</h2>
            <p style="font-size: 16px;">
                This project analyzes NYC property sales using regression models.<br>
                Two different multivariate regressions were run, one using Total Price and one using Total Units.<br>
                The goal of the project was to successfully create, test, train, and then cross-validate a regression model.
            </p>
            <br>
            <p style="font-size: 16px;">
                All data was sourced from 
                <a href="https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu/about_data" 
                   target="_blank" style="color: #1f77b4;">NY Open Data</a>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
###_____________________________________________________________________________
### Opens the Exploratory Analysis and Cleaning section using HTML for formatting
elif section == "Exploratory Analysis and Cleaning":
    st.subheader("Exploratory Analysis and Cleaning")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Introduction</h2>
            <p style='font-size:16px;'>
            Data was first accessed from the NY OpenData website. Using the website API token, batches were downloaded in 50,000 row chunks.<br>
            Then using libraries within Python, 17 different CSV files were combined into one mass file. The 17 other files were then discarded.<br><br>
            Following downloading, the data cleaning was conducted to ensure that the dataset was ready for analysis and a regression.<br><br>
            Columns were first viewed using the Pandas .info() function. Then within each column null values were investigated and dropped.<br>
            The feature "boroughs" was originally a numeric code, it was remapped to categorical values.<br>
            A new feature was then engineered, Units per Square Feet Percentage was created by total units divided by gross square feet.<br>
            An additional feature was also engineered, floor area ratio was computed by using gross square feet / land square feet.<br>
            A rolling average was created for residential units by computing averages based off of borough and year built.<br>
            These averages were used to populate missing data within the residential unit column.<br><br>
            The complete data set can be seen below:
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("#### Preview of Cleaned DataFrame:")
    st.dataframe(df, height=400)
### Above code adds a preview window of the cleaned data frame
###_____________________________________________________________________________
###Opens section for the total price regression
elif section == "Regression - Total Price":
    st.subheader("Regression - Total Price")
    
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Introduction</h2>
            <p style='font-size:16px;'>
            Beginning with Total Price as the target variable, two different scalers were applied:<br>
            one using a log transformation and one using a log base 10.
            <br>
            <br>
            The features included were: total units, residential units, gross square feet, land square feet, year built,<br> 
            units per sqft pct, net square feet, borough Brooklyn, borough Manhattan, borough Queens, borough Staten Island, borough The Bronx”
            <br>
            <br>
            <br>
            Prior to the regression being run a visualization of the log base 10 and the price were run to see the distribution. Please see below: 
            <br>
            <br>
          

          </p>
        </div>
        """,
        unsafe_allow_html=True
    
    
    )
        ### Local variable calculation. Ensures that values are greater than 0
    df['log10_sale_price'] = np.where(df['sale_price'] > 0, np.log10(df['sale_price']), np.nan)
    df = df.dropna(subset=['log10_sale_price'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ### Original Sale Price Histogram
    sns.histplot(df['sale_price'] / 100000, bins=100, ax=axes[0], edgecolor='black', color='steelblue')
    axes[0].set_title("Original Sale Price Distribution")
    axes[0].set_xlabel("Sale Price (in hundreds of thousands)")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}K'))

    #### Log10 Sale Price Histogram
    sns.histplot(df['log10_sale_price'], bins=60, ax=axes[1], edgecolor='black', color='orange')
    axes[1].set_title("Log10-Transformed Sale Price Distribution")
    axes[1].set_xlabel("log₁₀(Sale Price)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_ylim(0, 1200)

    ### Display the plot in Streamlit
    st.pyplot(fig)

    st.markdown(
      """ 
      <div style='text-align: center;'>
      <h2>Results</h2>
      <p style='font-size:16px;'>
      Once the Regression was run the following R² variables were returned: 
        """,
        unsafe_allow_html=True
    
    
    )
    ### Uses previously saved dataframe to display the output of the R² metrics 
    st.markdown("### R² Comparison by Target Transformation")
    st.table(r2_df.reset_index(drop=True).style.hide(axis="index"))
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Commentary</h2>
            <p style='font-size:18px;'>
            It can be noted that the R² values returned by this regression fail to explain the variance in the dependent variable given the independent variables.<br>
            The correlation returned by the independent variables is very low. Log scaling the target variable actually decreased the R². <br>
            It can also be seen that the overall correlation is less for the log scaled target. <br> 
        

          </p>
        </div>
        """,
        unsafe_allow_html=True
    
    
    )
    ### Calculate numeric subset
    df_numeric = df.select_dtypes(include=["float64", "int64"]).copy()
    df_numeric = df_numeric[df_numeric['sale_price'] > 0]

    ### Ensure log10_sale_price exists
    df_numeric['log10_sale_price'] = np.where(df_numeric['sale_price'] > 0, np.log10(df_numeric['sale_price']), np.nan)

    ### Sale price correlation
    exclude_1 = ['sale_price_100k', 'log_sale_price'] 
    df_corr_1 = df_numeric.drop(columns=[col for col in exclude_1 if col in df_numeric.columns], errors='ignore')
    corr_matrix_1 = df_corr_1.corr()
    corr_target_1 = corr_matrix_1[["sale_price"]].sort_values(by="sale_price", ascending=False).head(10)

    ### Log10 sale price correlation
    exclude_2 = ['sale_price_100k', 'log_sale_price']
    df_corr_2 = df_numeric.drop(columns=[col for col in exclude_2 if col in df_numeric.columns], errors='ignore')
    corr_matrix_2 = df_corr_2.corr()
    corr_target_2 = corr_matrix_2[["log10_sale_price"]].sort_values(by="log10_sale_price", ascending=False).head(10)

    ### Plot side-by-side in one figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ### Plot 1: Sale Price
    sns.heatmap(corr_target_1, annot=True, cmap="coolwarm", vmin=-1, vmax=1,
                ax=axes[0], annot_kws={"size": 8})
    axes[0].set_title("Correlation with Sale Price", fontsize=10)
    axes[0].tick_params(labelsize=8)
    axes[0].set_ylabel("")
    axes[0].tick_params(axis='x', labelrotation=45)

    ### Plot 2: Log10 Sale Price
    sns.heatmap(corr_target_2, annot=True, cmap="coolwarm", vmin=-1, vmax=1,
                ax=axes[1], annot_kws={"size": 8})
    axes[1].set_title("Correlation with Log₁₀ Sale Price", fontsize=10)
    axes[1].tick_params(labelsize=8)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    st.pyplot(fig)

###_____________________________________________________________________________
###Open section for Regression Total Units


elif section == "Regression - Total Units":
    st.subheader("Regression - Total Units")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Commentary</h2>
            <p style='font-size:16px;'>
            A second regression was conducted using Total Units as the target independent variable. <br> 
            The dependent variables were: Gross Square Feet, Land Square Feet, Year Built, Sales Price, Units per Square Feet (percentage), net square feet, floor area ratio, and the categorical borough variables.<br>
            Due to over all higher correlation between the target variable and features a much higher R² value was achieved, in both training and testing. 
        

          </p>
        </div>
        """,
        unsafe_allow_html=True
    
    
    )

    st.markdown("### Regression Performance Metrics")

    ### Format numbers and add dataframe for training versus testing
    formatted_df = metrics_df.copy()
    formatted_df["Training"] = formatted_df["Training"].round(4)
    formatted_df["Testing"] = formatted_df["Testing"].round(4)

    st.dataframe(formatted_df.reset_index(drop=True), use_container_width=True)

    st.subheader("Regression - Total Units")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Interactive Regression</h2>
            <p style='font-size:16px;'>
            Please see interactive regression below to see estimated Total Units given specific independent variables. 

          </p>
        </div>
        """,
        unsafe_allow_html=True

    )
    ### Adds the interactive regression window with the associated variables to streamlit. 
    st.markdown("## Estimate Total Units")
    ### imports the saved regression model from the other workbook using the library Joblib
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("multiple_linear_model.pkl")

    ### Adds user inputs with mins and maxes for each of the feature. 
    land_sqft = st.number_input("Land Square Feet", min_value=100.0, max_value=1_000_000.0, value=500.0, step = 1.0, key = "land_sqft")
    gross_default = min(1000.0, float(land_sqft))
    gross_sqft = st.number_input("Gross Square Feet (Building foot print) - cannot exceed land square feet value", min_value=100.0, max_value=float(land_sqft), value=gross_default, step = 1.0, key="gross_default")
    if gross_sqft > land_sqft:
        gross_sqft = land_sqft
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    sale_price = st.number_input("Sale Price ($)", min_value=10_000.0, max_value=1_000_000_000.0, value=1_000_000.0)
    net_sqft = max(0.0, float(land_sqft) - float(gross_sqft))
    st.number_input("Net Square Feet (auto = Land − Gross)",
    value=float(net_sqft), step=1.0, disabled=True, key="net_sqft_readonly")
    #units_per_sqft_pct = st.slider("Units per Sqft (as %)", min_value=0.0, max_value=100.0, value=50.0)
    #floor_area_ratio = st.slider("Floor Area Ratio", min_value=0.0, max_value=12.0, value=2.5)
    UNITS_PER_SQFT_CONST = 59.62   
    FAR_CONST             = 2.0

    units_per_sqft_pct = UNITS_PER_SQFT_CONST
    floor_area_ratio   = FAR_CONST

    ### Codes the borough based on the user response. 1 if the borough is picked 0 otherwise.
    borough = st.selectbox("Select Borough", ["Brooklyn", "Manhattan", "Queens", "Staten Island", "The Bronx"])
    borough_brooklyn = 1 if borough == "Brooklyn" else 0
    borough_manhattan = 1 if borough == "Manhattan" else 0
    borough_queens = 1 if borough == "Queens" else 0
    borough_staten_island = 1 if borough == "Staten Island" else 0
    borough_the_bronx = 1 if borough == "The Bronx" else 0
    ### Adds feature inputs into an array
    input_features = np.array([[ 
        gross_sqft,
        land_sqft,
        year_built,
        sale_price,
        units_per_sqft_pct,
        net_sqft,
        borough_brooklyn,
        borough_manhattan,
        borough_queens,
        borough_staten_island,
        borough_the_bronx,
        floor_area_ratio
        
    ]])

    input_scaled = scaler.transform(input_features)
    predicted_units = model.predict(input_scaled)[0]
    ### Output of the predicted units based on the user inputs. 
    st.markdown(f"<h4 style='text-align: center;'>Predicted Total Units: {predicted_units:.2f}</h4>", unsafe_allow_html=True)
    ### Ends the interactive section 

    ### Adds a bit of commentary on the R² value in HTML formatted style. 
    st.markdown(
    """
    <div style='text-align: center;'>
        <h2>Commentary</h2>
        <p style='font-size:16px;'>
        Due to the higher R² value it can be seen that this regression model is better at explaining the proportion of variance in the dependent variable given the independent variable.
        <br>
        <br>
      </p>
    </div>
    """,
    unsafe_allow_html=True

    )
    ### plots a line of best fit masking valus up to a threshold of 1000 to keep the graph neater. 
    threshold = 1000
    mask = (y_all < threshold) & (y_pred_all < threshold)
    y_actual_clean = y_all[mask]
    y_pred_clean = y_pred_all[mask]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(y_actual_clean, y_pred_clean, alpha=0.6)
    ax.plot([0, max(y_actual_clean.max(), y_pred_clean.max())],
            [0, max(y_actual_clean.max(), y_pred_clean.max())],
            color='red', linestyle='--', label='Perfect Prediction')
    ax.set_xlabel("Actual Total Units", fontsize=8)
    ax.set_ylabel("Predicted Total Units", fontsize=8)
    ax.set_title("Actual vs Predicted Total Units", fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
###_____________________________________________________________________________
### Opens the conclusion section using HTML to format.
elif section == "Conclusion":
    st.subheader("Conclusion")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Commentary</h2>
            <p style='font-size:16px;'>
            When performing any kind of analysis, it is important to remember that the analysis can only be as good as the data that was presented. Through a lengthy cleaning
            and engineering process. It was determined that given Total Price, the chosen features within this dataset were not a good predictors of price. 
            Based off this analysis the major reason features were not a statistically significant  indicator of price is due to the lack of correlation. 
            <br>
            <br> 
            In regards to Total Units, the chosen independent variables offered a much high-level of correlation resulting in a higher R² value, in both training and testing. While the R²
            value fit the requirements of the project it is understood that the value is too close to 1. Implying potential multicollinearity or potential overfitting. 
            <br>
            The variables that were engineered for this regression may be to similar to the target variable of Total Units. The model could be also be "overfit", learning the training data to well. 
            Based  the increase in R² value between training and testing there is reasonable enough evidence to conclude overfitting. 
            
        

          </p>
        </div>
        """,
        unsafe_allow_html=True
    
    
    )
### Adds the full correlation matrixs with all variables to the end of the conclusion section.
    correlation_matrix = df.corr(numeric_only=True)
    st.subheader("Correlation Matrix")
    st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm').format(precision=2))


###_____________________________________________________________________________
### Opens the introduction to project 2 using HTML format 
elif section == "Introduction - Project 2 Classification":
    st.subheader("Introduction - Project 2 Classification")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Multi-Model Classification Analysis</h2>
            <p style='font-size:16px;'>
            During the next section of this analysis a variation of the same data set will be used to complete multiple different kinds of classification analyses.
            6 different types of classification models will be used. The goal is to test which model is the most accurate at predicting the price level of a given building based on the borough it's in.
            The price levels are: Low, Medium, High, and Very High. Price levels are determined by the borough that the building is housed in. So each different borough has a different threshold for each given price level.
            The price level variable was engineered so that each bucket has 25% of the total 2024 sales per borough within it. 
            <br>
            <br>
            <h2>AI Agent</h2>
            <p style='font-size:16px;'>
            With the improvement and accessibility  of LLMs, the rise of AI agents has increased and become more user friendly and easier to deploy. 
            <br>
            For the purpose of this project an AI Agent has been developed to provide restaurant recommendations based off of the current weather and location. 
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
###_____________________________________________________________________________
### Opens Classification Intro
elif section == "Multi-Classification Models":
    st.subheader("Multi-Classification Models")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Data Visuals</h2>
            <p style='font-size:16px;'>
            Please view three different visuals of the 2024 NYC housing data set. 

          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Visual 1: Monthly sale volume by borough (2024)")

    ### Ensures sale_date is datetime
    df_2024['sale_date'] = pd.to_datetime(df_2024['sale_date'], errors='coerce')

    ### Map borough IDs to names
    BOROUGH_MAP = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'}
    df_2024['borough_name'] = (
      pd.to_numeric(df_2024['borough'], errors='coerce')
        .map(BOROUGH_MAP)
        .fillna(df_2024['borough'].astype(str))
    )

    ### Filter to 2024 only
    d24 = df_2024[df_2024['sale_date'].dt.year == 2024].copy()

    ### Aggregate monthly counts by borough_name
    monthly_counts = (
      d24.groupby([d24['sale_date'].dt.to_period('M').astype(str), 'borough_name'])
          .size()
          .reset_index(name='count')
          .rename(columns={'sale_date': 'month'})
    )

    ### Build a complete 12-month list for x-axis
    months = pd.period_range('2024-01', '2024-12', freq='M').astype(str)
    borough_order = sorted(monthly_counts['borough_name'].unique())

    ### Plots
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(
        data=monthly_counts,
        x='month', y='count',
        hue='borough_name', hue_order=borough_order,
        marker='o', markersize=4, linewidth=1.8,
        ax=ax
    )

    ## Legend outside on the right (cleanest for dashboards)
    ax.legend(
        title='Borough',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    ### X axis: 12 months, rotated neatly
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=30, ha='right', fontsize=9)

    ax.set_xlabel('Month of Sale')
    ax.set_ylabel('Number of Sales')
    ax.set_title('Monthly Sales Volume by Borough (2024)')

    ### Make room for the outside legend and lower x-labels
    plt.subplots_adjust(right=0.82, bottom=0.22)
    st.pyplot(fig)

    st.markdown("### Visual 2: Price Distribution by Borough (2024)")

    if 'borough_order' not in locals():
      borough_order = sorted(df_2024['borough_name'].dropna().unique())

    ### Keep positive prices only changes into log10. Y-Scale remains in USD (Not log 10.) 
    df_box = df_2024[df_2024['sale_price'] > 0].copy()
    df_box['log_sale_price'] = np.log10(df_box['sale_price'])

    sns.set_style("whitegrid")

    ### One subplot per borough, shared Y so scales match
    fig, axes = plt.subplots(
      1, len(borough_order),
      figsize=(25, 6),
      sharey=True
    )

    for ax, b in zip(axes, borough_order):
      sns.boxplot(
          data=df_box[df_box['borough_name'] == b],
          y='log_sale_price',
          ax=ax
      )
      ax.set_title(b)
      ax.set_xlabel("")
      ### Only the first subplot gets the y-axis label
      ax.set_ylabel('Sale Price (Log10)' if ax is axes[0] else '')

    ### Y-axis ticks as dollar powers of 10 (10^3 to 10^9)
    yticks = range(3, 10)  
    for ax in axes:
      ax.set_yticks(yticks)
      ax.set_yticklabels([f"${10**y:,.0f}" for y in yticks])

    plt.tight_layout()
    st.pyplot(fig)


    st.markdown("### Visual 3: Property Count by Price Category per Borough (2024)")


    ### Ensure price_category is a string and ordered properly
    df_2024['price_category'] = df_2024['price_category'].astype(str)
    cat_order = CategoricalDtype(
        categories=['Low', 'Medium', 'High', 'Very High'],
        ordered=True
    )
    df_2024['price_category'] = df_2024['price_category'].astype(cat_order)

    ### Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df_2024,
        x='borough_name',
        hue='price_category',
        hue_order=cat_order.categories,
        palette='Set2',
        ax=ax
    )

    ax.set_title('Property Count by Price Category per Borough (2024)')
    ax.set_xlabel('Borough')
    ax.set_ylabel('Number of Properties')
    ax.legend(title='Price Category')

    plt.tight_layout()
    st.pyplot(fig)
###_____________________________________________________________________________
### Opens the Model Results Section
elif section == "Model Results":
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Model Results</h2>
            <p style='font-size:16px;'>
            Seven different models were deployed for this analysis: Logistic regression, naive bayes, decision trees, random forests, support vector machines (SVM),K-nearest Neighbors (KNN), and K means (Unsupervised).
            For the purpose of this analysis, all models asides from K means will be compared to each other. K-Means is excluded from the comparison because it is an unsupervised. output labels are arbitrary clusters rather than the learned class labels.
            For K-mean results please view the end of the page.  
            <br>
            For supervised learning, the following metrics are used to evaluate each model: accuracy, prevision, recall, and f1-score. 
            <br>
            For unsupervised learning, the metrics of evaluation will be Adjusted Random Index (ARI) & Normalized Mutual Information (NMI).  
            <br>
            <br>
            <br>
            Please see model summary below: 

          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    metrics_df = pd.read_excel("model_results.xlsx")
    st.dataframe(metrics_df)

    st.markdown(
      """
      <div style='text-align: center;'>
        <p style='font-size:16px;'>
        It can be seen that out of the 6 different supervised models the Random Forrest model is the winning supervised model with the highest accurary of 65.5%. 
        <br>
        <br> 
        In regards to the unsupervised K - Means test, it can be noted that the model did not return any real useful information. <br> 
        Using an optimal K =7 the ARI was 0.0090 and the MNI was 0.0174.

      </div>
      """,
      unsafe_allow_html=True
    )
 ###____________________________________________________________________________
 ### Opens the AI Agent Section 
elif section == "AI Agent":
  st.subheader("Step 1: Check the weather for a neighborhood/City (Most locations can be used -- Distances Measured from location center)")

  ### Creates rules for recommendations based on the weather. Returns a list of foods that fit the weather and a rationale of why they fit. 
  ### This function is inplace incase the AI does not work. 
  def cuisine_recs(weather: dict):
      """
      Return (picks, rationale) based on weather dict with keys:
      temp_f, precip_in, wind_mph, description
      """
      desc = (weather.get("description") or "").lower()
      temp = float(weather.get("temp_f") or 70)
      precip = float(weather.get("precip_in") or 0)

      ### Defines the rules using controls. 
      if "rain" in desc or "shower" in desc or precip > 0.05:
          return (
              ["Soup", "Ramen", "Pho", "Indian"],
              "Rainy weather—cozy soups and warm, comforting dishes hit the spot."
          )
      elif temp >= 85:
          return (
              ["Salad", "Sushi", "Mediterranean", "Smoothies", "Ice Cream"],
              "Hot day—light and refreshing options are best."
          )
      elif temp <= 50:
          return (
              ["Ramen", "Hotpot", "Curry", "BBQ", "Italian"],
              "Chilly weather—warming broths and hearty comfort food are ideal."
          )
      else:
          return (
              ["Pizza", "Tacos", "Burgers", "Brunch", "Mediterranean"],
              "Mild day—plenty of patio-friendly crowd-pleasers to choose from."
          )
  ### this takes the totoal dataframe and creates a short list from it. 
  def shortlist_for_ai(df, top_n: int = 6) -> str:
      """Compact bullet list for the AI prompt: 'Name — cuisine (0.4 mi)'."""
      import pandas as pd
      if df is None or (isinstance(df, pd.DataFrame) and df.empty):
          return "No restaurants found."
      rows = []
      for _, r in df.head(top_n).iterrows():
          name = r.get("name", "Unknown")
          cuisine = r.get("cuisine", "") or ""
          dist = r.get("distance_miles", "")
          rows.append(f"- {name} — {cuisine} ({dist} mi)")
      return "\n".join(rows)
    

  neighborhood = st.text_input("Neighborhood / Area (e.g., 'SoHo, Manhattan, NY')")
  ### Initiates when the weather button is pressed. 
  if st.button("Get Weather"):
      if not neighborhood.strip():
          st.warning("Please enter a neighborhood.")
      else:
          try:
              with st.spinner("Geocoding…"):
                  lat, lon = geocode_place(neighborhood)
                  st.session_state.latlon = (lat, lon)  ### cache for later steps
              with st.spinner("Fetching weather…"):
                  wx = get_weather(lat, lon)
                  st.session_state.wx = wx           ### cache for later steps

              st.success(f"Weather for {neighborhood}")
              st.write(f"**Conditions:** {wx['description']}")
              st.write(f"**Temperature:** {wx['temp_f']} °F")
              st.write(f"**Precipitation:** {wx['precip_in']} in")
              st.write(f"**Wind:** {wx['wind_mph']} mph")
          except Exception as e:
              st.error(f"Sorry, something went wrong: {e}")

  st.subheader("Step 2: Find nearby restaurants")
  ### Adds a slider that lets the user pick distances. 
  radius_miles = st.slider("Search radius (miles)", 0.10, 1.00, 0.75, 0.05)
  ### Initates whent he restuarant button is pressed. 
  if st.button("Find Restaurants"):
      if not neighborhood.strip():
        ### Controls in place so that the previous two buttons have to be presseed before this one. 
          st.warning("Please enter a neighborhood above first.")
      else:
          try:
              ### reuse cached geocode if available
              if "latlon" in st.session_state:
                  lat, lon = st.session_state.latlon
              else:
                  with st.spinner("Geocoding…"):
                      lat, lon = geocode_place(neighborhood)
                      st.session_state.latlon = (lat, lon)

              with st.spinner("Searching restaurants…"):
                  df_rest = overpass_restaurants(lat, lon, radius_miles=radius_miles)
                  st.session_state.df_rest = df_rest  ### keep for Step 3

              if df_rest.empty:
                  st.info("No restaurants found in this radius. Try a larger radius (up to 1 mile).")
              else:
                  st.success(f"Found {len(df_rest)} restaurants within {radius_miles:.2f} miles")
                  st.dataframe(df_rest[["name", "cuisine", "distance_miles", "osm_url"]])

                  ### quick map (if lat/lon columns exist)
                  try:
                      st.map(df_rest.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
                  except Exception:
                      pass

          except Exception as e:
              st.error(f"Sorry, something went wrong: {e}")

  st.subheader("Step 3: Weather-smart AI summary")

  if st.button("Generate AI Summary"):
      if not neighborhood.strip():
          st.warning("Please enter a neighborhood above first.")
          st.stop()

      ### Make sure we have restaurants from Step 2
      if "df_rest" not in st.session_state or st.session_state.df_rest is None or st.session_state.df_rest.empty:
          st.error("Please run Step 2 (Find Restaurants) first.")
          st.stop()

      df_rest = st.session_state.df_rest

      ### Ensure weather is in place (reuse if saved; otherwise fetch)
      if "wx" in st.session_state:
          wx = st.session_state.wx
      else:
          ### reuse cached geocode if available
          if "latlon" in st.session_state:
              lat, lon = st.session_state.latlon
          else:
              with st.spinner("Looking up weather…"):
                  lat, lon = geocode_place(neighborhood)
                  st.session_state.latlon = (lat, lon)
          with st.spinner("Fetching weather…"):
              wx = get_weather(lat, lon)
              st.session_state.wx = wx

      ### Weather-aware cuisine picks
      picks, rationale = cuisine_recs(wx)

      ###Build shortlist text for the AI prompt
      shortlist = shortlist_for_ai(df_rest, top_n=6)

      st.write(f"**Weather:** {wx['description']} · {wx['temp_f']}°F · {wx['precip_in']} in · {wx['wind_mph']} mph")
      st.write(f"**Weather-aware cuisine picks:** {', '.join(picks) if picks else 'Any'}")
      st.caption(rationale)

      ### Ask the hosted model (OpenRouter); fallback if key missing/unavailable
      try:
          with st.spinner("Asking the AI guide…"):
                summary = ai_summary_with_openai(neighborhood, wx, picks, shortlist, model="gpt-3.5-turbo")
            st.success("AI Summary")
            st.write(summary)
      except Exception as e:
        st.warning("AI service unavailable — showing a quick summary instead.")
        st.caption(f"{type(e).__name__}: {e}") 
    
