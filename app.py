import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Indian Flight Delay Analysis",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------
# LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("/content/india_flight_delay_model_input.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found! Please place "
        "india_flight_delay_model_input.csv "
        "inside your project folder."
    )
    st.stop()

# --------------------------------
# WEBSITE HEADER
# --------------------------------

st.title("✈️ Indian Flight Delay Analysis")
st.markdown(
    "### Exploratory Data Analysis Dashboard"
)

st.write(
    "This dashboard explores flight delays "
    "and the factors that may influence them."
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Dataset Explorer",
        "Delay Analysis",
        "Correlation Analysis"
    ]
)

# --------------------------------
# OVERVIEW
# --------------------------------

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flights", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

    if "Delay_Target" in df.columns:

        st.subheader("Flight Delay Distribution")

        counts = df["Delay_Target"].value_counts().sort_index()

        fig, ax = plt.subplots()

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Flight Delay Target Distribution")
        ax.set_xlabel("Delay Target")
        ax.set_ylabel("Number of Flights")

        st.pyplot(fig)

        delay_rate = (
            df["Delay_Target"].mean() * 100
        )

        st.metric(
            "Delay Target Rate",
            f"{delay_rate:.2f}%"
        )

# --------------------------------
# DATASET EXPLORER
# --------------------------------

elif page == "Dataset Explorer":

    st.header("Dataset Explorer")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(info, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

# --------------------------------
# DELAY ANALYSIS
# --------------------------------

elif page == "Delay Analysis":

    st.header("Flight Delay Analysis")

    possible_columns = [
        col for col in
        ["Airline", "Origin_Airport", "Destination_Airport"]
        if col in df.columns
    ]

    if possible_columns:

        selected_column = st.selectbox(
            "Select a category",
            possible_columns
        )

        if "Delay_Target" in df.columns:

            analysis = (
                df.groupby(selected_column)["Delay_Target"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )

            st.subheader(
                f"Delay Rate by {selected_column}"
            )

            st.bar_chart(analysis)

        else:
            st.warning("Delay_Target column is not available.")

    else:
        st.info(
            "Airline and airport category columns "
            "were not found in this dataset."
        )

# --------------------------------
# CORRELATION ANALYSIS
# --------------------------------

elif page == "Correlation Analysis":

    st.header("Correlation Analysis")

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            annot=False,
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)

    else:
        st.warning(
            "Not enough numerical columns for correlation analysis."
        )

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Indian Flight Delay Analysis | Exploratory Data Analysis Project"
)