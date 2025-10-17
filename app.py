import streamlit as st
import polars as pl
import plotly.express as px
import warnings
import os
import time

def replace_dollar_with_fullwidth(text):
    """Replace dollar signs with fullwidth equivalents to prevent LaTeX rendering"""
    if isinstance(text, str):
        return text.replace('$', '＄')
    return text

@st.cache_data
def load_data():
    """Load the Titanic dataset using Polars"""
    return pl.read_csv("data/titanic.csv.gz")

def suppress_deprecation_warnings():
    """Suppress specific deprecation warnings from appearing in the Streamlit UI"""
    warnings.filterwarnings(
        "ignore",
        message=(
            "The keyword arguments have been deprecated and will be removed in a future release. "
            "Use config instead to specify Plotly configuration options."
        ),
        category=DeprecationWarning,
    )

def render_plotly_chart(fig):
    """Render Plotly figures using only the supported `config` parameter.
    Avoids deprecated keyword arguments entirely so no banner appears.
    """
    return st.plotly_chart(fig, config={"responsive": True})

def show_image_if_exists(path, width="stretch"):
    """Show an image if the file exists; otherwise, show a non-blocking warning."""
    if os.path.exists(path):
        st.image(path, width=width)
    else:
        st.warning(f"Image not found: {path}")

def display_basic_info(df):
    """Display basic dataset information"""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Passengers", df.height)
    
    with col2:
        st.metric("Total Columns", df.width)
    
    with col3:
        size_kb = df.estimated_size() / 1024
        st.metric("Memory Usage", f"{size_kb:.1f} KB")
    
    st.subheader("Dataset Schema")
    st.write(df.schema)
    
    # Lazy filter controls
    st.subheader("All Rows")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        _cols = list(df.columns)
        _default_idx = _cols.index("Name") if "Name" in _cols else 0
        filter_column = st.selectbox("Filter column", options=_cols, index=_default_idx)
    with col_f2:
        filter_value = st.text_input("Filter value (substring match)", value="")
        case_sensitive = st.checkbox("Case sensitive match", value=False)

    if filter_column and filter_value and filter_column in df.columns:
        series_expr = pl.col(filter_column).cast(pl.Utf8)
        if case_sensitive:
            predicate = series_expr.str.contains(filter_value, literal=True)
        else:
            predicate = series_expr.str.to_lowercase().str.contains(filter_value.lower(), literal=True)
        lazy_df = df.lazy().filter(predicate)
        start_filter = time.perf_counter()
        shown_df = lazy_df.collect()
        end_filter = time.perf_counter()
        st.metric("Filter time (seconds)", f"{end_filter - start_filter:.6f}")
        st.write(f"Filtered rows: {shown_df.height}")
        st.dataframe(shown_df.to_pandas(), width="stretch")
    else:
        st.metric("Filter time (seconds)", f"{0.0:.6f}")
        st.dataframe(df.to_pandas(), width="stretch")

def analyze_passenger_sex(df):
    """Analyze passenger sex distribution"""
    st.subheader("👥 Passenger Sex Analysis")
    
    # Value counts
    sex_counts = df["Sex"].value_counts()
    sex_proportions = (
        sex_counts.with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .select([pl.col("Sex"), pl.col("proportion")])
        .sort("proportion", descending=True)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Count by Sex:**")
        st.dataframe(sex_counts.to_pandas(), width="stretch")
    
    with col2:
        st.write("**Proportion by Sex:**")
        st.dataframe(sex_proportions.to_pandas(), width="stretch")
    
    # Bar chart
    fig = px.bar(
        sex_counts.to_pandas().reset_index(),
        x="Sex",
        y="count",
        title="Number of Passengers by Sex",
        color="Sex",
        color_discrete_map={"male": "#1f77b4", "female": "#ff7f0e"}
    )
    fig.update_layout(showlegend=False)
    render_plotly_chart(fig)

def analyze_passenger_classes(df, show_survival_rate_image=False):
    """Analyze passenger classes and survival rates"""
    st.subheader("🏛️ Passenger Class Analysis")
    
    # Add classes image
    show_image_if_exists("img/titanic_classes.jpg", width=400)
    
    # Pivot table for survival by class
    survival_by_class = df.pivot(
        values="PassengerId",
        index="Pclass",
        on="Survived",
        aggregate_function="len"
    )
    
    st.write("**Passenger Count by Class and Survival:**")
    st.dataframe(survival_by_class.to_pandas(), width="stretch")
    
    # Survival rates by class
    survival_rates = df.group_by("Pclass").agg(
        pl.col("Survived").mean().alias("Survival_Rate")
    ).sort("Pclass")
    
    # if show_survival_rate_image:
    #     show_image_if_exists("img/titanic_door.jpg", width=400)
    st.write("**Survival Rate by Class:**")
    st.dataframe(survival_rates.to_pandas(), width="stretch")
    
    # Bar chart for survival rates
    fig = px.bar(
        survival_rates.to_pandas(),
        x="Pclass",
        y="Survival_Rate",
        title="Survival Rate by Passenger Class",
        labels={"Pclass": "Passenger Class", "Survival_Rate": "Survival Rate"}
    )
    fig.update_layout(yaxis_tickformat=".1%")
    render_plotly_chart(fig)

def analyze_embarkation_ports(df):
    """Analyze passenger embarkation ports"""
    st.subheader("🚢 Embarkation Port Analysis")
    
    # Add route image
    show_image_if_exists("img/titanic_route.png", width=400)
    
    # Get unique ports
    unique_ports = df["Embarked"].unique()
    st.write("**Unique Ports:**")
    st.dataframe(pl.DataFrame({"Embarked": unique_ports.to_list()}).to_pandas(), width="stretch")
    
    # Replace abbreviations with full names
    df_ports = df.with_columns(
        pl.col("Embarked").replace({
            "C": "Cherbourg",
            "S": "Southampton", 
            "Q": "Queenstown"
        })
    )
    
    # Count passengers by port
    port_counts = df_ports["Embarked"].value_counts()
    port_proportions = (
        df_ports["Embarked"].value_counts()
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .select([pl.col("Embarked"), pl.col("proportion")])
        .sort("proportion", descending=True)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Passengers by Port:**")
        st.dataframe(port_counts.to_pandas(), width="stretch")
    
    with col2:
        st.write("**Proportion by Port:**")
        st.dataframe(port_proportions.to_pandas(), width="stretch")
    
    # Pie chart
    fig = px.pie(
        port_counts.to_pandas().reset_index(),
        values="count",
        names="Embarked",
        title="Passenger Distribution by Embarkation Port"
    )
    render_plotly_chart(fig)

def analyze_passenger_names(df):
    """Analyze passenger names and find the captain"""
    st.subheader("👨‍✈️ Passenger Name Analysis")
    
    # Add captain image
    show_image_if_exists("img/titanic_captain.jpg", width=400)
    
    # Extract titles from names
    titles = (df['Name'].str.split(", ")
              .list.reverse()
              .list.join(" ")
              .str.split(".")
              .list.get(0)
              .value_counts()
              .sort("count", descending=True))
    
    st.write("**Passenger Titles Distribution:**")
    st.dataframe(titles.head(10).to_pandas(), width="stretch")
    
    # Create bar chart for passenger titles
    fig = px.bar(
        titles.head(10).to_pandas().reset_index(),
        x="Name",
        y="count",
        title="Passenger Titles Distribution",
        labels={"Name": "Title", "count": "Number of Passengers"},
        color="count",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    render_plotly_chart(fig)
    
    # Find the captain
    captain = df.filter(pl.col("Name").str.contains(", Capt."))
    
    if captain.height > 0:
        st.write("**Titanic Captain Information:**")
        captain_info = captain.select("Name", "Age", "Pclass")
        st.dataframe(captain_info.to_pandas(), width="stretch")
    else:
        st.write("No captain found in the dataset")

def create_scatter_plots(df):
    """Create scatter plots for age vs fare analysis"""
    st.subheader("📈 Age vs Fare Analysis")
    
    # Get min and max values for sliders
    age_series = df["Age"].drop_nulls()
    fare_series = df["Fare"].drop_nulls()
    if age_series.len() > 0:
        min_age = int(age_series.min())
        max_age = int(age_series.max())
    else:
        min_age = 0
        max_age = 0
    if fare_series.len() > 0:
        min_fare = float(fare_series.min())
        max_fare = float(fare_series.max())
    else:
        min_fare = 0.0
        max_fare = 0.0
    
    # Create sliders for filtering
    col1, col2 = st.columns(2)
    
    with col1:
        age_range = st.slider(
            "Select Age Range:",
            min_value=min_age,
            max_value=max_age,
            value=(max(min_age, 18), max_age),
            step=1,
            help="Filter passengers by age range"
        )
    
    with col2:
        fare_range = st.slider(
            "Select Fare Range:",
            min_value=min_fare,
            max_value=min(max_fare, 300.0),
            value=(min_fare, min(max_fare, 300.0)),
            step=1.0,
            help="Filter passengers by fare range"
        )
    
    # Filter data based on slider values
    filtered_passengers = df.filter(
        (pl.col("Age") >= age_range[0]) & 
        (pl.col("Age") <= age_range[1]) &
        (pl.col("Fare") >= fare_range[0]) & 
        (pl.col("Fare") <= fare_range[1])
    )
    
    st.write(replace_dollar_with_fullwidth(f"**Filtered Passengers (Age: {age_range[0]}-{age_range[1]}, Fare: ${fare_range[0]:.2f}-${fare_range[1]:.2f}):** {filtered_passengers.height} passengers"))
    
    # Scatter plot with class colors
    fig = px.scatter(
        filtered_passengers.to_pandas(),
        x="Age",
        y="Fare",
        color="Pclass",
        title="Titanic Passengers - Interactive Filtering",
        labels={"Pclass": "Passenger Class"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    render_plotly_chart(fig)
    
    # Age distribution by class (using filtered data)
    fig2 = px.box(
        filtered_passengers.to_pandas(),
        x="Pclass",
        y="Age",
        title="Age Distribution by Passenger Class (Filtered)",
        labels={"Pclass": "Passenger Class", "Age": "Age"}
    )
    render_plotly_chart(fig2)

def analyze_survival_by_sex_and_class(df):
    """Analyze survival rates by sex and class"""
    st.subheader("💀 Survival Analysis by Sex and Class")
    
    # Show door image above survival charts
    show_image_if_exists("img/titanic_door.jpg", width=400)
    
    # Create survival rate table
    survival_table = (df.rename({"Pclass": "Passenger Class"})
                     .with_columns(Sex=pl.col("Sex").str.to_titlecase())
                     .pivot(values='Survived',
                            index='Passenger Class',
                            on='Sex',
                            aggregate_function='mean')
                     .sort("Passenger Class"))
    
    st.write("**Survival Rate by Sex and Class:**")
    st.dataframe(survival_table.to_pandas(), width="stretch")
    
    # Create heatmap
    survival_df = survival_table.to_pandas().set_index('Passenger Class')
    fig = px.imshow(
        survival_df.values,
        x=survival_df.columns,
        y=survival_df.index,
        title="Survival Rate Heatmap (Sex vs Class)",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    fig.update_layout(
        xaxis_title="Sex",
        yaxis_title="Passenger Class",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['1', '2', '3']
        )
    )
    render_plotly_chart(fig)

def main():
    """Main Streamlit application"""
    suppress_deprecation_warnings()
    st.set_page_config(
        page_title="Titanic Data Analysis with Polars",
        page_icon="🚢",
        layout="wide"
    )
    
    st.title("🚢 Titanic Dataset Analysis with Polars")
    
    # Add main Titanic image
    show_image_if_exists("img/titanic.jpg", width=400)
    
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return
    
    # Create sidebar for navigation
    # st.sidebar.title("📋 Navigation")
    analysis_type = st.sidebar.pills(
        "**Choose Analysis Type:**",
        options=[
            "Dataset Overview",
            "Passenger Sex Analysis", 
            "Passenger Class Analysis",
            "Embarkation Port Analysis",
            "Passenger Name Analysis",
            "Age vs Fare Analysis",
            "Survival Analysis",
            "All Analyses"
        ],
        selection_mode="single",
        default="Dataset Overview",
        width="stretch"
    )
    
    if analysis_type == "Dataset Overview" or analysis_type == "All Analyses":
        display_basic_info(df)
    
    if analysis_type == "Passenger Sex Analysis" or analysis_type == "All Analyses":
        analyze_passenger_sex(df)
    
    if analysis_type == "Passenger Class Analysis" or analysis_type == "All Analyses":
        analyze_passenger_classes(df, show_survival_rate_image=(analysis_type == "All Analyses"))
    
    if analysis_type == "Embarkation Port Analysis" or analysis_type == "All Analyses":
        analyze_embarkation_ports(df)
    
    if analysis_type == "Passenger Name Analysis" or analysis_type == "All Analyses":
        analyze_passenger_names(df)
    
    if analysis_type == "Age vs Fare Analysis" or analysis_type == "All Analyses":
        create_scatter_plots(df)
    
    if analysis_type == "Survival Analysis" or analysis_type == "All Analyses":
        analyze_survival_by_sex_and_class(df)

    
    # Footer
    st.markdown("**Data Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)")
    st.markdown("**Powered by:** [aifab.xyz](https://aifab.xyz) + Polars + Streamlit + Plotly")
    st.markdown("**GitHub:** [truevis/aifab-titanic](https://github.com/truevis/aifab-titanic)")

if __name__ == "__main__":
    main()
