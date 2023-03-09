import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from pathlib import Path

FILEPATH = Path(__file__).parents[1]

# Read dataset
df = pd.read_csv(FILEPATH / 'data/original.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Read Image
image = Image.open('pics/logo_transparent.png')


# Create a histrical graph
def show_historical(ticker):
    st.markdown("")
    st.markdown("")
    st.subheader("Histrical prices are...")

    fig = px.line(df, x='date', y=ticker, title='Price : ' + ticker + ' (USD)')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)


# Section to show each prophet
def show_prophet():
    st.markdown("")
    st.markdown("")
    st.subheader("Our Prophets are...")

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(':violet[NBeats]')
        st.image("pics/owl.jpeg")

    with col2:
        st.markdown(':green[Kalman]')
        st.image("pics/monkey.jpeg")

    with col3:
        st.markdown(':red[ARIMA]')
        st.image("pics/download.jpeg")

    with col4:
        st.markdown(':blue[Meta]')
        st.image("pics/cat.jpeg")

    with col5:
        st.markdown(':orange[T.F.T.]')
        st.image("pics/bear.jpeg")

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")


# Create a prediction graph
def show_prediction(ticker):
    st.markdown("")
    st.markdown("")
    st.subheader("Their predictions are...")

    fig = px.line(df, x='date', y=ticker, title='Price:' + ticker)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)


# Area for each recommendation
def recommend(currency, model):

    if 3 - 1 > 2:
        st.markdown(':red[Sell!]')
    elif 3 - 1 < 2:
        st.markdown(':green[Buy!]')
    else:
        st.markdown('Hold!')


def show_advice(ticker):
    st.markdown("")
    st.markdown("")
    st.subheader("Their positions are...")
    st.markdown("")
    st.markdown("")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(':violet[NBeats]')
        recommend(df[ticker], 1)

    with col2:
        st.markdown(':green[Kalman]')
        recommend(df[ticker], 2)

    with col3:
        st.markdown(':red[ARIMA]')
        recommend(df[ticker], 3)

    with col4:
        st.markdown(':blue[Meta]')
        recommend(df[ticker], 4)

    with col5:
        st.markdown(':orange[T.F.T.]')
        recommend(df[ticker], 5)


# Create Space...
def create_space():
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("***")


# Explanation of Models...
def explain_model():
    st.markdown("How does each prophet predict prices?")
    st.markdown("")
    st.markdown("")

    with st.expander("ARIMA -> ARIMA"):
        st.markdown(
            "ARIMA stands for AutoRegressive Integrated Moving Average. ")
        st.markdown(
            "It is a time series forecasting model that is used to predict future values of a stationary time series based on its past values.")
        st.markdown("")
        st.markdown("The ARIMA model has three components:")
        st.markdown("1. Autoregression (AR): This refers to the process of using past values of a time series to predict future values. The AR component of the model uses the past values of the series to forecast its future values.")
        st.markdown("2. Integrated (I): This refers to the process of differencing the time series to make it stationary. Stationarity is an important property of time series data for many statistical models, including ARIMA. The I component of the model makes the time series stationary.")
        st.markdown("3. Moving Average (MA): This refers to the process of using the past errors of the time series to predict its future values. The MA component of the model uses the past errors to forecast the future values.")

    with st.expander("Meta -> Facebook's Prophet"):
        st.markdown("The Prophet model is a time series forecasting model developed by Facebook. It is designed to be highly scalable, accurate, and easy to use, making it an ideal tool for forecasting time series data.")
        st.markdown("")
        st.markdown(
            "The Prophet model uses a decomposable time series model with three main components:")
        st.markdown(
            "1. Trend: This component models non-periodic changes in the time series.")
        st.markdown(
            "2. Seasonality: This component models periodic changes in the time series.")
        st.markdown(
            "3. Holidays: This component models the impact of holidays and other special events on the time series.")

    with st.expander("Kalman -> Kalman Filter"):
        st.markdown("The Kalman filter is a mathematical algorithm used to estimate the state of a system based on noisy and incomplete measurements. It was developed by Rudolf Kalman in the 1960s and has been widely used in fields such as engineering, economics, and physics.")
        st.markdown("")
        st.markdown("The basic idea behind the Kalman filter is to use a set of mathematical equations to estimate the state of a system based on a series of measurements taken over time. The filter works by taking the most recent estimate of the state of the system, combining it with new measurements, and then using this updated information to make a new estimate of the state. This process is repeated over time, with each new estimate incorporating both the current measurement and the previous estimate of the state.")
        st.markdown("")
        st.markdown("The Kalman filter is particularly useful in situations where the measurements of a system are noisy or incomplete, such as in tracking the position of a moving object or in controlling a robot. By using a probabilistic model of the system and its measurements, the filter is able to make accurate estimates of the state even in the presence of noise and uncertainty.")

    with st.expander("Nbeats -> N-BEATS"):
        st.markdown("The N-BEATS algorithm is a deep learning approach to time series forecasting, developed by Oreshkin et al. in 2020. The algorithm is designed to be scalable and adaptable, and it has achieved state-of-the-art performance on several benchmark time series forecasting datasets.")
        st.markdown("")
        st.markdown('The N-BEATS algorithm works by decomposing a time series into a set of "trend" and "seasonality" components, which are then forecasted separately and recombined to produce the final forecast. The algorithm is based on a deep neural network architecture, which is trained to learn the optimal decomposition of the time series and the corresponding forecasts.')
        st.markdown("")
        st.markdown('The N-BEATS architecture consists of a stack of fully connected layers, with each layer processing a different "horizon" of the time series (i.e., a different length of time into the future). The layers are arranged in a hierarchical fashion, with lower layers processing shorter horizons and higher layers processing longer horizons. This allows the algorithm to capture both short-term and long-term trends in the data.')

    with st.expander("T.F.T -> Temporal Fusion Transformer"):
        st.markdown("The Temporal Fusion Transformer (TFT) is a deep learning approach to time series forecasting, introduced by Bryan Lim et al. in 2019. The TFT is designed to model complex, non-linear relationships between time series inputs and outputs, and it has achieved state-of-the-art performance on several benchmark time series forecasting datasets.")
        st.markdown("")
        st.markdown("At a high level, the TFT works by transforming the input time series into a set of features that can be used to make a forecast. The algorithm is based on a deep neural network architecture that combines the transformer and autoregressive models to model the temporal relationships between the input and output time series")
        st.markdown("")
        st.markdown("The TFT architecture consists of several key components, including an encoder that transforms the input time series into a set of features, a decoder that generates the forecast, and a temporal attention mechanism that allows the model to capture the dependencies between the input and output time series.")


# Rendering
def render():
    st.image(image)
    st.header('Our five prophets predict the Future of Cryptocurrency...')

    create_space()

    # Create tabs for cryptocurrencies
    tickers = df.columns[:-1]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        list(df.columns[:-1]))

    with tab1:
        show_historical(tickers[0])
        show_prophet()
        show_prediction(tickers[0])
        show_advice(tickers[0])

    with tab2:
        show_historical(tickers[1])
        show_prophet()
        show_prediction(tickers[1])
        show_advice(tickers[1])

    with tab3:
        show_historical(tickers[2])
        show_prophet()
        show_prediction(tickers[2])
        show_advice(tickers[2])

    with tab4:
        show_historical(tickers[3])
        show_prophet()
        show_prediction(tickers[3])
        show_advice(tickers[3])

    with tab5:
        show_historical(tickers[4])
        show_prophet()
        show_prediction(tickers[4])
        show_advice(tickers[4])

    with tab6:
        show_historical(tickers[5])
        show_prophet()
        show_prediction(tickers[5])
        show_advice(tickers[5])

    with tab7:
        show_historical(tickers[6])
        show_prophet()
        show_prediction(tickers[6])
        show_advice(tickers[6])

    with tab8:
        show_historical(tickers[7])
        show_prophet()
        show_prediction(tickers[7])
        show_advice(tickers[7])

    create_space()
    explain_model()

    create_space()

    # Disclaimer
    st.markdown(
        'The content of this webpage is not an investment advice and does not constitute any offer or solicitation to offer or recommendation of any investment product. '
        'It is for general purposes only and does not take into account your individual needs, investment objectives and specific financial circumstances. '
        'Investment involves risk.'
    )


render()
