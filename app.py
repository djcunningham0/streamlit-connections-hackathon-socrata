import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
import plotly.express as px
from plotly.subplots import make_subplots

from connections import SocrataConnection
import utils


st.set_page_config(
    page_title="City of Chicago Datasets",
    page_icon="./images/chicago_flag.png",
)

st.title("City of Chicago Datasets")
st.write("---")

conn = st.experimental_connection("socrata", type=SocrataConnection)


PAGES = {
    0: "explore datasets",
    1: "COVID-19 vaccination rates",
    2: "speed cameras",
}

with st.sidebar:
    st.image("./images/chicago_flag.png")
    page = st.radio("Select page:", PAGES.keys(), format_func=lambda x: PAGES[x])

if page == 0:
    st.write("## Explore datasets")
    st.write("""
        This page uses the Socrata connection to explore the available City of
        Chicago datasets. First it pulls all available datasets using the connection,
        then it embeds all of the dataset descriptions using a transformer model
        (specifically [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).
        When the user enters a query, the app executes a semantic search using the
        to return the most relevant datasets. The "show sample data" button in each 
        search result uses the Socrata connection to pull a sample from the dataset.
    """)
    st.write("---")

    datasets = conn.client().datasets()
    datasets = [x for x in datasets if x["resource"]["type"] == "dataset"]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [f"ID: {x['resource']['id']}; title: {x['resource']['name']}; description: {x['resource']['description']}"
              for x in datasets]

    @st.cache_data(ttl=3600)
    def embed_corpus() -> Tensor:
        return embedder.encode(corpus, convert_to_tensor=True)

    corpus_embeddings = embed_corpus()

    c1, c2 = st.columns(2)
    query = c1.text_input("query", placeholder="traffic crashes",
                          help="""Enter a query (for example, "traffic crashes" or "business licenses")
                          and then search to retrieve related datasets.""")
    query = query.strip()
    n = c2.number_input("top n", value=10, min_value=1, max_value=50,
                        help="The number of results to return")

    if query != "":
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores: Tensor = util.cos_sim(query_embedding, corpus_embeddings)[0]

        idx = cos_scores.argsort(descending=True)
        colors = utils.create_evenly_spaced_colors(n)
        for i in range(n):
            st.write(utils.shadow_hr("top", color=colors[i]), unsafe_allow_html=True)
            st.markdown(utils.dataset_card(datasets[idx[i]]), unsafe_allow_html=True)
            if st.button("show sample data", key=f'button_{datasets[idx[i]]["resource"]["id"]}'):
                df = conn.get(datasets[idx[i]]["resource"]["id"], limit=5)
                st.dataframe(df)
            st.write(utils.shadow_hr("bottom", color=colors[i]), unsafe_allow_html=True)

if page == 1:
    st.write("## COVID-19 vaccination rates")
    st.write("""
            This page demos how data can be pulled using the Socrata connection
            and then used to create visualizations. Specifically, this page 
            explores two datasets related to COVID-19 vaccinations in the city 
            of Chicago. First we use the 
            [COVID-19 Vaccinations by ZIP Code](https://data.cityofchicago.org/Health-Human-Services/COVID-19-Vaccinations-by-ZIP-Code/553k-3xzc) 
            dataset to create an interactive map of vaccination rate by zip
            code over time. Then we use the 
            [COVID-19 - Vaccinations by Region, Age, and Race-Ethnicity](https://data.cityofchicago.org/Health-Human-Services/COVID-19-Vaccinations-by-Region-Age-and-Race-Ethni/n7f2-e2kq)
            dataset to create line charts showing how vaccination rates have 
            changed over time split by zone, age, and race.
        """)
    st.write("---")

    val = st.selectbox("choose which percentage to show", ["1st dose %", "All doses %", "Booster %"],
                       help="This dropdown affects all charts on the page.")

    st.write("### Vaccination status by zip code over time")
    # read in the COVID Vaccinations by Zip Code dataset
    df = conn.get_all("553k-3xzc").sort_values("date").reset_index(drop=True)
    numeric_cols = [
        "_1st_dose_percent_population",
        "vaccine_series_completed_percent_population",
        "bivalent_percent_population",
        "population",
    ]
    for x in numeric_cols:
        df[x] = df[x].astype(float).fillna(0)

    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[~df["zip_code_location"].isna()]
    df["lat"] = df["zip_code_location"].apply(lambda x: x["coordinates"][1])
    df["lon"] = df["zip_code_location"].apply(lambda x: x["coordinates"][0])

    # filter the dataframe to include only the dates closest to the start of each month
    df = utils.filter_to_first_entry_of_month(df, group_cols="zip_code")

    df = df.rename(columns={
        "_1st_dose_percent_population": "1st dose %",
        "vaccine_series_completed_percent_population": "All doses %",
        "bivalent_percent_population": "Booster %",
    })

    fig = (
        px.scatter_mapbox(
            data_frame=df,
            lat="lat",
            lon="lon",
            animation_frame="month",
            size="population",
            color=val,
            mapbox_style="stamen-terrain",
            color_continuous_scale=px.colors.diverging.Portland_r,
            range_color=[0, 1],
            zoom=8.5,
            height=600,
            hover_data={
                "zip_code": True,
                "lat": False,
                "lon": False,
                "month": False,
                "population": ":,",
                val: ":.1%",
            },
        )
        .update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0})
    )
    st.plotly_chart(fig)

    st.write("### Vaccination status by zone, age, and race")
    # read in the COVID-19 Vaccinations by Region, Age, and Race-Ethnicity dataset
    df = conn.get_all("n7f2-e2kq").sort_values("date").reset_index(drop=True)
    numeric_cols = [
        "_1st_dose",
        "vaccine_series_completed",
        "bivalent",
        "population_size",
    ]
    for x in numeric_cols:
        df[x] = df[x].astype(float)

    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[~df["region_centroid"].isna()]
    df["lat"] = df["region_centroid"].apply(lambda x: x["coordinates"][1])
    df["lon"] = df["region_centroid"].apply(lambda x: x["coordinates"][0])

    df = df.rename(columns={
        "equity_zone": "City Zone",
        "age_group": "Age",
        "race_ethnicity": "Race",
    })

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    for group_col in ["City Zone", "Age", "Race"]:
        plot_df = (
            df.copy()
            .groupby(["date", group_col], as_index=False)
            .agg({
                "_1st_dose": "sum",
                "vaccine_series_completed": "sum",
                "bivalent": "sum",
                "population_size": "sum",
            })
        )

        plot_df = plot_df.loc[plot_df[group_col] != "Unknown"]

        # Other, non-Latinx group doesn't sum to 1 for some reason
        if group_col == "Race":
            plot_df = plot_df.loc[plot_df["Race"] != "Other, non-Latinx"]

        for x in ["_1st_dose", "vaccine_series_completed", "bivalent"]:
            plot_df[f"{x}_pct"] = plot_df[x] / plot_df["population_size"] * 100

        # filter the dataframe to include only the dates closest to the start of each month
        plot_df = utils.filter_to_first_entry_of_month(plot_df, group_cols=group_col)

        plot_df = plot_df.rename(columns={
            "_1st_dose_pct": "1st dose %",
            "vaccine_series_completed_pct": "All doses %",
            "bivalent_pct": "Booster %",
        })

        fig = (
            px.line(
                plot_df,
                x="date",
                y=val,
                color=group_col,
                height=300,
                labels=dict(date=""),
                hover_data={
                    "date": False,
                    val: ":.1f",
                }
            )
            .update_yaxes(ticksuffix="%")
            .update_yaxes(range=[0, 100])
            .update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 10})
        )
        st.plotly_chart(fig)


elif page == 2:
    st.write("## Speed camera violations")
    st.write("""
                This page demos how data can be pulled using the Socrata connection
                and then used to create visualizations. Specifically, this page 
                explores the [Speed Camera Violations](https://data.cityofchicago.org/d/hhkd-xvj4) 
                dataset to display an interactive map of speed camera violations
                throughout the city, as well as a 
            """)
    st.write("---")

    # load speed camera violations
    df = conn.get_all("hhkd-xvj4", where="violation_date >= '2019-01-01'")
    for x in ["latitude", "longitude", "violations"]:
        df[x] = df[x].astype(float)
    df["violation_date"] = pd.to_datetime(df["violation_date"])

    min_date = min(df["violation_date"])
    max_date = max(df["violation_date"])
    start_date, end_date = st.slider(
        "select date range",
        min_value=min_date.date(),
        value=[min_date.date(), max_date.date()],
        max_value=max_date.date(),
        format="YYYY-MM-DD",
    )

    plot_df = (
        df
        .loc[df["violation_date"].dt.date >= start_date]
        .loc[df["violation_date"].dt.date <= end_date]
        .copy()
        .groupby(["camera_id", "address", "latitude", "longitude"], as_index=False)
        .agg({
            "violations": "sum",
            "violation_date": ["min", "max"],
        })
    )

    # flatten column multiindex
    plot_df.columns = [col[0] if col[1] == "" else "_".join(col).strip() for col in plot_df.columns.values]
    plot_df = plot_df.rename(columns={"violations_sum": "violations"})

    # calculate number of days each camera has been in operation
    plot_df["n_days"] = (plot_df["violation_date_max"] - plot_df["violation_date_min"]).dt.days
    plot_df["violations_per_day"] = plot_df["violations"] / plot_df["n_days"]

    fig = px.scatter_mapbox(
        plot_df,
        lat="latitude",
        lon="longitude",
        size="violations",
        mapbox_style="stamen-terrain",
        zoom=8.5,
        height=600,
        hover_data={
            "address": True,
            "camera_id": True,
            "violations": True,
            "violations_per_day": ":.2f",
            "latitude": False,
            "longitude": False,
        },
    )
    st.plotly_chart(fig)

    df["week_start_date"] = df["violation_date"] - pd.to_timedelta(df["violation_date"].dt.dayofweek, unit="D")
    plot_df = (
        df
        .groupby(["week_start_date"], as_index=False)
        .agg({
            "violations": "sum",
            "camera_id": "nunique",
        })
    )

    st.write("""
        *Note:* The threshold for ticketing was reduced from 10 to 6 MPH over
        the speed limit in March 2021 (dotted red line).
    """)
    vertical_line_date = "2021-03-01"
    fig = (
        px.line(
            plot_df,
            x="week_start_date",
            y="violations",
            hover_data={"week_start_date": ":%Y-%m-%d"}
        )
        .add_shape(
            type="line",
            x0=vertical_line_date,
            x1=vertical_line_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dot")
        )
    )
    st.plotly_chart(fig)
