from get_data import get_data
from predict import predict

df = get_data()

pred_arima = predict(df)

print(pred_arima)
