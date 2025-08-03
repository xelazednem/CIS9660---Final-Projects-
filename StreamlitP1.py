#!/usr/bin/env python
# coding: utf-8
###Library Imports 
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import joblib
import seaborn as sns
###_________________________________________________________________________________

### Streamlit Heading#####
st.set_page_config(page_title="Project 1- Regression", layout="wide")
st.set_page_config(page_title="Project 1 - Regression", layout="wide")
###________________________________________________________________________________
### Import necessary datasets
df = pd.read_csv("Streamlit_dataset.csv")

###______________________________________________________________
# Sidebar navigation
st.set_page_config(page_title="Project 1- Regression", layout="wide")
st.set_page_config(page_title="Project 1 - Regression", layout="wide")
# Sidebar navigation
section = st.sidebar.radio("Navigation", [
    "Introduction",
    "Exploratory Analysis and Cleaning",
    "Regression - Total Price",
    "Regression - Total Units",
    "Conclusion",
])

### Global Variables_____________________________________________________________

### Total Price Metrics
r2_raw = 0.0927
r2_log = 0.0049
r2_log10 = 0.0048
r2_df = pd.DataFrame({
    "Target Transformation": ["Raw Price", "Log(Price)", "Log₁₀(Price)"],
    "R² Score": [r2_raw, r2_log, r2_log10]
})


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
if section == "Introduction":
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
    gross_sqft = st.number_input("Gross Square Feet", min_value=100.0, max_value=1_000_000.0, value=1000.0)
    land_sqft = st.number_input("Land Square Feet", min_value=100.0, max_value=1_000_000.0, value=500.0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    sale_price = st.number_input("Sale Price ($)", min_value=10_000.0, max_value=1_000_000_000.0, value=1_000_000.0)
    units_per_sqft_pct = st.slider("Units per Sqft (as %)", min_value=0.0, max_value=100.0, value=50.0)
    net_sqft = st.number_input("Net Square Feet", min_value=100.0, max_value=1_000_000.0, value=900.0)
    floor_area_ratio = st.slider("Floor Area Ratio", min_value=0.0, max_value=12.0, value=2.5)

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
            When performing any kind of analysis, it is important to remember that the analysis can only be as good as the data that was presented. Through a length cleaning
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


