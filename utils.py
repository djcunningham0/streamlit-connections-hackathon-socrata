from datetime import datetime
import colorsys
from typing import Union, List

import pandas as pd


def dataset_card(dataset_dict: dict, max_desc_length: int = 300) -> str:
    dataset_id = dataset_dict["resource"]["id"]
    name = dataset_dict["resource"]["name"]
    description = dataset_dict["resource"]["description"]
    permalink = dataset_dict["permalink"]
    updated_date = (
        datetime.strptime(dataset_dict["resource"]["updatedAt"], "%Y-%m-%dT%H:%M:%S.%fZ")
        .strftime("%Y-%m-%d")
    )
    category = dataset_dict["classification"]["domain_category"]

    if len(description) > max_desc_length:
        description = description[:max_desc_length] + f'... <a href="{permalink}" target="_blank">See more</a>'

    html_code = f"""
            <h4 style="margin: 0;"><a href="{permalink}" target="_blank">{name}</a></h4>
            <hr style="margin-top: 0; margin-bottom: 20px;">
            <div style="display: flex; padding: 0;">
                <p style="flex: 1; text-align: center"><i>Dataset ID:</i><br>{dataset_id}</p>
                <p style="flex: 1; text-align: center"><i>Category:</i><br>{category}</p>
                <p style="flex: 1; text-align: center;"><i>Last updated:</i><br>{updated_date}</p>
            </div>
            <hr style="margin-top: 0; margin-bottom: 20px;">
            <p>{description}</p>
            <hr style="margin-top: 0; margin-bottom: 20px;">
        """
    return html_code


def shadow_hr(direction: str, color: str = "rgba(256, 256, 256)") -> str:
    if direction == "bottom":
        shadow = f"box-shadow: inset 0 -12px 12px -12px {color}"
        margin = "margin-top: 10px;"
    else:
        shadow = f"box-shadow: inset 0 12px 12px -12px {color}"
        margin = "margin-bottom: 10px;"
    return f"""
    <hr style="height: 15px; border: 0; {shadow}; {margin};">
    """


def create_evenly_spaced_colors(n: int):
    colors = []
    for i in range(n):
        hue = i / n  # Distribute hues evenly between 0 and 1
        saturation = 0.8  # You can adjust this value if needed (0.0 to 1.0)
        lightness = 0.5  # You can adjust this value if needed (0.0 to 1.0)
        rgba = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgba = tuple(int(val * 255) for val in rgba)
        colors.append(f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]})")
    return colors


def filter_to_first_entry_of_month(df: pd.DataFrame, group_cols: Union[str, List[str]], date_col: str = "date") -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    df["day_difference"] = df[date_col].dt.day
    df = (
        df
        .groupby([df[date_col].dt.year, df[date_col].dt.month, *group_cols])
        .apply(lambda x: x.loc[abs(x["day_difference"] - 1).idxmin()])
        .reset_index(drop=True)
    )
    df["month"] = df[date_col].dt.strftime("%B %Y")
    return df
