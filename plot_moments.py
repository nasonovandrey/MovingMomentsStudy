import argparse

import matplotlib.pyplot as plt
import pandas as pd

from moving_moment import moving_moment

from weights import *  # isort: skip


def parse_weights(weights_args):
    """Parses the weight arguments into a list of tuples containing the weight
    function and its parameters.

    :param weights_args: List of strings containing weight function
        names and parameters
    :return: List of tuples in the form (function_name, radius, n)
    """
    parsed_weights = []
    i = 0
    while i < len(weights_args):
        func_name = weights_args[i]
        radius = int(weights_args[i + 1])
        n = int(weights_args[i + 2])
        parsed_weights.append((func_name, radius, n))
        i += 3
    return parsed_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Moving Moments of Time Series Data."
    )
    parser.add_argument("--file", required=True, help="Path to the data file.")
    parser.add_argument(
        "--column", required=True, help="Column name for the data to analyze."
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        help="Sequence of weight functions and their parameters.",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.file)

    column = args.column
    data = df[column].to_numpy()

    times = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 8))
    plt.plot(times, data, label="Original Time Series")

    weights_specs = parse_weights(args.weights)
    for func_name, radius, n in weights_specs:
        adjusted_times = times[radius:-radius]
        weight_func = eval(f"{func_name}")
        moving_moment_data = moving_moment(data, weight_func, radius, n)
        plt.plot(
            adjusted_times,
            moving_moment_data,
            label=f"Moving {n}th Moment ({func_name}, radius={radius})",
            linestyle="--",
        )

    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.title(f"Time Series and its Moving {n}th Moments")
    plt.legend()
    plt.show()
