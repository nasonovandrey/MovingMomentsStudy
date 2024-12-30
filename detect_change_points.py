import numpy as np
import ruptures as rpt


def detect_change_points(dataframe, column):
    values = dataframe[column].values.reshape(-1, 1)

    model = "l2"

    algo = rpt.Pelt(model=model, min_size=1).fit(values)

    penalty = np.log(len(values)) * np.std(values) ** 2

    result = algo.predict(pen=penalty)

    change_points_dates = dataframe["date"][result[:-1]].tolist()
    return change_points_dates
