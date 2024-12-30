import argparse

import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS

from detect_change_points import detect_change_points
from moving_moment import moving_moment

from weights import *  # isort: skip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Granger test on moving moments.")
    parser.add_argument("--file", required=True, help="Path to the data file.")
    parser.add_argument(
        "--column", required=True, help="Column name for the data to analyze."
    )
    parser.add_argument(
        "--radius", type=int, default=10, required=False, help="Window's raidus."
    )
    parser.add_argument(
        "--weight", required=True, help="Weight function for moving moments."
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.file)

    column = args.column

    change_points = detect_change_points(df, column)
    all_points = [df["date"].min()] + change_points + [df["date"].max()]
    intervals = [(all_points[i], all_points[i + 1]) for i in range(len(all_points) - 1)]

    data = df[column].to_numpy()
    weight_func = args.weight

    radius = args.radius
    model_results = []

    for start_date, end_date in intervals:
        segment = df[(df["date"] > start_date) & (df["date"] <= end_date)]

        moments_data = {"moment_1": [], "moment_2": [], "moment_3": [], "moment_4": []}

        for n in range(1, 5):
            moments_data[f"moment_{n}"] = moving_moment(
                segment[column], eval(weight_func), radius, n
            )

        for key in moments_data.keys():
            moments_data[key] = np.array(moments_data[key])

        moments_df = pd.DataFrame(moments_data)
        future_close = segment[column].shift(-radius)

        # Now, ensure 'moments_df' is created from the segment with aligned index
        # Note: Adjust this part to ensure moments_data contains arrays of correct length
        # For now, let's assume moving_moment function returns aligned data
        moments_df = pd.DataFrame(moments_data, index=segment.index[radius:-radius])  # Adjust index as needed

        # Assign 'future_close' correctly, ensuring alignment by using loc or similar to match indices
        moments_df = moments_df.loc[future_close.index]  # Align moments_df to the same index as future_close
        moments_df["future_close"] = future_close

        # Now proceed to drop NaN values, ensuring alignment remains correct
        data_for_model = moments_df.dropna()

        X = data_for_model[["moment_1", "moment_2", "moment_3", "moment_4"]]
        y = data_for_model["future_close"]
        X = sm.add_constant(X)

        model = OLS(y, X).fit()

        model_results.append((start_date, end_date, model))

    for start_date, end_date, model in model_results:
        print(f"Interval {start_date} to {end_date}:")
        print(model.summary())
        print("-" * 50)
