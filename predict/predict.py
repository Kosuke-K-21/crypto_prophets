import numpy as np
from darts import TimeSeries as ts
from darts.models.forecasting import arima
from darts.metrics import rmse


def preprocessing(df):
    tickers = list(df.columns).remove('date')

    series = ts.from_dataframe(df, "date", tickers)

    # calculate log difference
    series_ld = series.map(np.log).diff()

    val_length = 30

    train, val = series_ld[:-val_length], series_ld[-val_length:]

    return train, val, tickers


def predict_arima(df):
    train, val, tickers = preprocessing(df)

    model_arima = arima.ARIMA()

    potential_params = {
        'p': [0, 1, 2, 4, 6, 8, 10],
        'd': range(0, 3),
        'q': range(0, 3)
    }

    best_model_arima, _, _ = model_arima.gridsearch(
        potential_params,
        series=train[tickers[0]],
        val_series=val[tickers[0]],
        metric=rmse
    )

    best_model_arima.fit(train[tickers[0]])

    prediction_arima = best_model_arima.predict(len(val), num_samples=1000)

    prediction_arima_val = prediction_arima.map(
        lambda x: np.exp(np.add.accumulate(x)) * df.iat[-len(val), 0])

    return prediction_arima_val
