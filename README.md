## Titanic Streamlit App — How it uses Polars

This app analyzes the Titanic dataset using the Polars DataFrame library for fast, expressive data operations, while Streamlit and Plotly handle the UI and charts. Below is a concise tour of where and how Polars is used in `app.py`.

### Why Polars here
- **Speed and low memory use**: Columnar engine with vectorized operations.
- **Lazy execution where useful**: Build queries first, evaluate only when needed.
- **Ergonomic expressions**: Clear, chainable APIs for grouping, pivoting, filtering, and string ops.

### Data loading
- `pl.read_csv("data/titanic.csv.gz")` loads the gzipped CSV directly into a `pl.DataFrame`.
- Cached with Streamlit to avoid re-reading on reruns.

### Basic dataset info (Polars-native)
- **Shape and memory**: `df.height`, `df.width`, `df.estimated_size()`.
- **Schema**: `df.schema` displayed to users.

### Lazy filtering (performance-friendly demo)
- Builds a lazy plan with `df.lazy().filter(predicate)` where `predicate` is composed with `pl.col(...).str.*` expressions.
- Uses `.collect()` to execute and measure filter time for the current UI inputs.

### Value counts and proportions
- `df["Sex"].value_counts()` to get counts.
- Proportions derived via `.with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))` then sorted.

### Grouped aggregations
- Survival rate by class: `df.group_by("Pclass").agg(pl.col("Survived").mean().alias("Survival_Rate")).sort("Pclass")`.

### Pivot tables
- Class × survival counts: `df.pivot(values="PassengerId", index="Pclass", columns="Survived", aggregate_function="len")`.
- Survival heatmap base: `df.rename({...}).with_columns(...).pivot(..., aggregate_function="mean")`.

### String processing for names and titles
- Title extraction via list and string ops:
  - `pl.col("Name").str.split(", ").list.reverse().list.join(" ").str.split(".").list.get(0).value_counts()`.

### Mapping and categorical cleanup
- Embarkation codes to names using `with_columns(pl.col("Embarked").replace({"C": "Cherbourg", ...}))`.
- Unique port values via `df["Embarked"].unique()`.

### Range filtering for visuals
- UI-driven filters apply a Polars expression:
  - `(pl.col("Age") >= a0) & (pl.col("Age") <= a1) & (pl.col("Fare") >= f0) & (pl.col("Fare") <= f1)` followed by `df.filter(...)`.

### Interop with Plotly/Streamlit
- Charts and tables are rendered from Pandas objects: `df.to_pandas()` (or for derived frames) to feed Plotly and `st.dataframe`.
- Polars handles all computation; conversion happens only at the presentation boundary to keep UI libraries happy.

### Files of interest
- `app.py`: all Polars usage lives here alongside Streamlit/Plotly UI code.
- `data/titanic.csv.gz`: dataset loaded by Polars.

### Summary
- Polars powers data ingestion, schema inspection, filtering (eager and lazy), grouping, pivoting, value counting, string transforms, and categorical mapping.
- The app converts results to Pandas only for visualization, keeping compute in Polars for clarity and performance.

### Web App
App is live at https://aifab-titanic.streamlit.app/


### Attribution
This app was derived from the Polars tutorial repository: [datapythonista-talks/polars-tutorial](https://gitlab.com/datapythonista-talks/polars-tutorial)