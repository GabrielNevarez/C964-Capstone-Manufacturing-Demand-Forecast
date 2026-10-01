import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


DATA_PATH = "data/cleaned/Weekly_Product_Demand_Features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"
RESULTS_PATH = "data/cleaned/Model_Comparison.csv"


# Configure page
st.set_page_config(
    page_title="Manufacturing Demand Forecast",
    layout="wide"
)


# Load demand data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Week"] = pd.to_datetime(df["Week"])
    return df


# Load model results
@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


# Load forecasting model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model_results = load_results()
model = load_model()


# Model features
features = [
    "Prev_Week1",
    "Prev_Week2",
    "Prev_Week3",
    "Prev_Week4",
    "Rolling_Mean_4",
    "Month",
    "Week_of_Year"
]


# Create 4 week forecast
def create_multiweek_forecast(data, weeks=4):

    data = data.sort_values(
        "Week"
    ).copy()

    latest_rows = data.tail(4)

    if len(latest_rows) < 4:
        return None

    history = latest_rows[
        "Total_Demand"
    ].tolist()

    forecast_rows = []

    next_week = (
        data["Week"].max()
        + pd.Timedelta(weeks=1)
    )

    for _ in range(weeks):

        prev_week1 = history[-1]
        prev_week2 = history[-2]
        prev_week3 = history[-3]
        prev_week4 = history[-4]

        rolling_mean_4 = sum(
            history[-4:]
        ) / 4

        forecast_data = pd.DataFrame({
            "Prev_Week1": [prev_week1],
            "Prev_Week2": [prev_week2],
            "Prev_Week3": [prev_week3],
            "Prev_Week4": [prev_week4],
            "Rolling_Mean_4": [rolling_mean_4],
            "Month": [next_week.month],
            "Week_of_Year": [
                next_week.isocalendar().week
            ]
        })

        prediction = model.predict(
            forecast_data
        )[0]

        prediction = max(
            prediction,
            0
        )

        forecast_rows.append({
            "Week": next_week,
            "Predicted_Demand": prediction
        })

        history.append(
            prediction
        )

        next_week += pd.Timedelta(
            weeks=1
        )

    return pd.DataFrame(
        forecast_rows
    )


# Create 4 week total product forecast
def create_product_forecast(product_data, weeks=4):

    warehouse_forecasts = []

    warehouses = product_data[
        "Warehouse"
    ].unique()

    for warehouse in warehouses:

        warehouse_df = product_data[
            product_data["Warehouse"] == warehouse
        ].copy()

        forecast_df = create_multiweek_forecast(
            warehouse_df,
            weeks
        )

        if forecast_df is not None:

            forecast_df[
                "Warehouse"
            ] = warehouse

            warehouse_forecasts.append(
                forecast_df
            )

    if len(warehouse_forecasts) == 0:
        return None

    combined_forecast = pd.concat(
        warehouse_forecasts,
        ignore_index=True
    )

    total_forecast = (
        combined_forecast.groupby(
            "Week",
            as_index=False
        )["Predicted_Demand"]
        .sum()
    )

    return total_forecast


# Page title
st.title(
    "Manufacturing Demand Forecast"
)

st.write(
    "Review historical demand and generate "
    "four-week manufacturing demand forecasts."
)


# Forecast level
forecast_level = st.sidebar.radio(
    "Forecast Level",
    [
        "Product Total",
        "Product + Warehouse"
    ]
)


# Product selection
products = sorted(
    df["Product_Code"].unique()
)

selected_product = st.sidebar.selectbox(
    "Select Product",
    products
)


# Filter selected product
product_df = df[
    df["Product_Code"] == selected_product
].copy()


# Product total mode
if forecast_level == "Product Total":

    # Aggregate product demand across warehouses
    display_df = (
        product_df.groupby(
            "Week",
            as_index=False
        )["Total_Demand"]
        .sum()
    )

    selected_category = (
        product_df[
            "Product_Category"
        ]
        .iloc[0]
    )

    total_demand = display_df[
        "Total_Demand"
    ].sum()

    average_weekly_demand = display_df[
        "Total_Demand"
    ].mean()


    # Product information
    st.subheader(
        "Product Information"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Product",
        selected_product
    )

    col2.metric(
        "Category",
        selected_category
    )

    col3.metric(
        "Warehouses",
        product_df[
            "Warehouse"
        ].nunique()
    )


    # Product totals
    st.subheader(
        "Product Totals"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Historical Demand",
        f"{total_demand:,.0f} units"
    )

    col2.metric(
        "Average Weekly Demand",
        f"{average_weekly_demand:,.0f} units"
    )


    # Historical demand chart
    st.subheader(
        "Historical Weekly Demand"
    )

    historical_chart = display_df[
        [
            "Week",
            "Total_Demand"
        ]
    ].set_index(
        "Week"
    )

    st.line_chart(
        historical_chart
    )


    # Demand by warehouse
    st.subheader(
        "Demand by Warehouse"
    )

    warehouse_totals = (
        product_df.groupby(
            "Warehouse",
            as_index=False
        )["Total_Demand"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        warehouse_totals["Warehouse"],
        warehouse_totals["Total_Demand"]
    )

    ax.set_xlabel(
        "Warehouse"
    )

    ax.set_ylabel(
        "Total Demand"
    )

    ax.set_title(
        "Total Demand by Warehouse"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # Demand distribution
    st.subheader(
        "Demand Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.hist(
        display_df["Total_Demand"],
        bins=30
    )

    ax.set_xlabel(
        "Weekly Demand"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Weekly Demand Distribution"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # Product forecast
    forecast_df = create_product_forecast(
        product_df,
        4
    )


    if forecast_df is not None:

        four_week_total = forecast_df[
            "Predicted_Demand"
        ].sum()

        st.subheader(
            "4 Week Forecast"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Week 1 Forecast",
            f"{forecast_df.iloc[0]['Predicted_Demand']:,.0f} units"
        )

        col2.metric(
            "4 Week Total Forecast",
            f"{four_week_total:,.0f} units"
        )


        # Forecast table
        forecast_display = forecast_df.rename(
            columns={
                "Predicted_Demand":
                "Forecasted_Demand"
            }
        )

        forecast_display[
            "Forecasted_Demand"
        ] = forecast_display[
            "Forecasted_Demand"
        ].round(0)

        st.dataframe(
            forecast_display,
            hide_index=True
        )


        # Recent history and forecast chart
        st.subheader(
            "Historical and Forecasted Demand"
        )

        recent_history = display_df.tail(52)[
            [
                "Week",
                "Total_Demand"
            ]
        ].copy()

        recent_history = recent_history.rename(
            columns={
                "Total_Demand":
                "Actual Demand"
            }
        )

        recent_history[
            "Forecast"
        ] = None

        forecast_plot = forecast_df.copy()

        forecast_plot = forecast_plot.rename(
            columns={
                "Predicted_Demand":
                "Forecast"
            }
        )

        forecast_plot[
            "Actual Demand"
        ] = None

        combined_plot = pd.concat(
            [
                recent_history,
                forecast_plot
            ],
            ignore_index=True
        )

        combined_plot = combined_plot.set_index(
            "Week"
        )

        st.line_chart(
            combined_plot[
                [
                    "Actual Demand",
                    "Forecast"
                ]
            ]
        )


        # Manufacturing planning
        st.subheader(
            "Manufacturing Planning"
        )

        recent_average = display_df[
            "Total_Demand"
        ].tail(4).mean()

        first_forecast = forecast_df.iloc[
            0
        ]["Predicted_Demand"]

        if first_forecast > recent_average * 1.20:

            st.write(
                "Forecasted demand is more than 20% "
                "above the recent four-week average. "
                "Additional inventory or production "
                "capacity may need to be reviewed."
            )

        elif first_forecast < recent_average * 0.80:

            st.write(
                "Forecasted demand is more than 20% "
                "below the recent four-week average. "
                "Production and inventory levels may "
                "need to be reviewed."
            )

        else:

            st.write(
                "Forecasted demand is within 20% "
                "of the recent four-week average "
                "and indicates relatively stable "
                "short-term demand."
            )

    else:

        st.warning(
            "Not enough historical demand is available "
            "to generate a forecast."
        )


# Product and warehouse mode
else:

    warehouses = sorted(
        product_df[
            "Warehouse"
        ].unique()
    )

    selected_warehouse = st.sidebar.selectbox(
        "Select Warehouse",
        warehouses
    )


    # Filter warehouse
    display_df = product_df[
        product_df["Warehouse"]
        == selected_warehouse
    ].copy()

    display_df = display_df.sort_values(
        "Week"
    )


    selected_category = (
        display_df[
            "Product_Category"
        ]
        .iloc[0]
    )

    total_demand = display_df[
        "Total_Demand"
    ].sum()

    average_weekly_demand = display_df[
        "Total_Demand"
    ].mean()


    # Product information
    st.subheader(
        "Product Information"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Product",
        selected_product
    )

    col2.metric(
        "Warehouse",
        selected_warehouse
    )

    col3.metric(
        "Category",
        selected_category
    )


    # Demand totals
    st.subheader(
        "Warehouse Demand Totals"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Historical Demand",
        f"{total_demand:,.0f} units"
    )

    col2.metric(
        "Average Weekly Demand",
        f"{average_weekly_demand:,.0f} units"
    )


    # Historical demand
    st.subheader(
        "Historical Weekly Demand"
    )

    historical_chart = display_df[
        [
            "Week",
            "Total_Demand"
        ]
    ].set_index(
        "Week"
    )

    st.line_chart(
        historical_chart
    )


    # Demand by warehouse
    st.subheader(
        "Demand by Warehouse"
    )

    warehouse_totals = (
        product_df.groupby(
            "Warehouse",
            as_index=False
        )["Total_Demand"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        warehouse_totals["Warehouse"],
        warehouse_totals["Total_Demand"]
    )

    ax.set_xlabel(
        "Warehouse"
    )

    ax.set_ylabel(
        "Total Demand"
    )

    ax.set_title(
        "Total Demand by Warehouse"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # Demand distribution
    st.subheader(
        "Demand Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.hist(
        display_df["Total_Demand"],
        bins=30
    )

    ax.set_xlabel(
        "Weekly Demand"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Weekly Demand Distribution"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # Historical model predictions
    prediction_df = display_df.copy()

    prediction_df[
        "Predicted_Demand"
    ] = model.predict(
        prediction_df[
            features
        ]
    )

    prediction_df[
        "Predicted_Demand"
    ] = prediction_df[
        "Predicted_Demand"
    ].clip(
        lower=0
    )


    # Actual vs predicted
    st.subheader(
        "Actual vs Predicted Demand"
    )

    comparison_chart = prediction_df[
        [
            "Week",
            "Total_Demand",
            "Predicted_Demand"
        ]
    ].set_index(
        "Week"
    )

    comparison_chart = comparison_chart.rename(
        columns={
            "Total_Demand":
            "Actual Demand",
            "Predicted_Demand":
            "Predicted Demand"
        }
    )

    st.line_chart(
        comparison_chart
    )


    # Warehouse forecast
    forecast_df = create_multiweek_forecast(
        display_df,
        4
    )


    if forecast_df is not None:

        four_week_total = forecast_df[
            "Predicted_Demand"
        ].sum()


        # Product total forecast
        product_forecast_df = (
            create_product_forecast(
                product_df,
                4
            )
        )

        product_four_week_total = (
            product_forecast_df[
                "Predicted_Demand"
            ].sum()
        )


        st.subheader(
            "4 Week Forecast"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Week 1 Warehouse Forecast",
            f"{forecast_df.iloc[0]['Predicted_Demand']:,.0f} units"
        )

        col2.metric(
            "4 Week Warehouse Forecast",
            f"{four_week_total:,.0f} units"
        )

        col3.metric(
            "4 Week Product Forecast",
            f"{product_four_week_total:,.0f} units"
        )


        # Forecast table
        forecast_display = forecast_df.rename(
            columns={
                "Predicted_Demand":
                "Forecasted_Demand"
            }
        )

        forecast_display[
            "Forecasted_Demand"
        ] = forecast_display[
            "Forecasted_Demand"
        ].round(0)

        st.dataframe(
            forecast_display,
            hide_index=True
        )


        # Recent history and forecast chart
        st.subheader(
            "Historical and Forecasted Demand"
        )

        recent_history = display_df.tail(52)[
            [
                "Week",
                "Total_Demand"
            ]
        ].copy()

        recent_history = recent_history.rename(
            columns={
                "Total_Demand":
                "Actual Demand"
            }
        )

        recent_history[
            "Forecast"
        ] = None

        forecast_plot = forecast_df.copy()

        forecast_plot = forecast_plot.rename(
            columns={
                "Predicted_Demand":
                "Forecast"
            }
        )

        forecast_plot[
            "Actual Demand"
        ] = None

        combined_plot = pd.concat(
            [
                recent_history,
                forecast_plot
            ],
            ignore_index=True
        )

        combined_plot = combined_plot.set_index(
            "Week"
        )

        st.line_chart(
            combined_plot[
                [
                    "Actual Demand",
                    "Forecast"
                ]
            ]
        )


        # Manufacturing planning
        st.subheader(
            "Manufacturing Planning"
        )

        recent_average = display_df[
            "Total_Demand"
        ].tail(4).mean()

        first_forecast = forecast_df.iloc[
            0
        ]["Predicted_Demand"]

        if first_forecast > recent_average * 1.20:

            st.write(
                "Forecasted demand is more than 20% "
                "above the recent four-week average. "
                "Additional inventory or production "
                "capacity may need to be reviewed."
            )

        elif first_forecast < recent_average * 0.80:

            st.write(
                "Forecasted demand is more than 20% "
                "below the recent four-week average. "
                "Production and inventory levels may "
                "need to be reviewed."
            )

        else:

            st.write(
                "Forecasted demand is within 20% "
                "of the recent four-week average "
                "and indicates relatively stable "
                "short-term demand."
            )

    else:

        st.warning(
            "Not enough historical demand is available "
            "to generate a forecast."
        )


# Model performance
st.subheader(
    "Model Performance"
)

linear_results = model_results[
    model_results["Model"]
    == "Linear Regression"
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{linear_results['MAE']:,.2f}"
)

col2.metric(
    "RMSE",
    f"{linear_results['RMSE']:,.2f}"
)

col3.metric(
    "R²",
    f"{linear_results['R2']:.4f}"
)