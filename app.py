
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(
    page_title = "Stock Pricer App",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

alt.themes.enable("dark")

class Stock:
    def __init__(self, ticker, num_of_days, num_of_years, num_of_simulations):
        self.ticker = ticker
        self.num_of_days = num_of_days
        self.num_of_years = num_of_years
        self.num_of_simulations = num_of_simulations

    def __data__(self):
        '''
        using yfinance fetch data for the stock
        :return:
        '''
        self.stock_data = yf.download(self.ticker, period=f'{self.num_of_days}d', interval='1d')['Close']

    def __price__(self):
        '''
        simulate prices based Geometric Brownian Motion
        :return: numpy array of simulated prices
        '''
        self.log_returns = np.log(self.stock_data / self.stock_data.shift(1)).dropna()

        self.mean = np.mean(self.log_returns * 252)
        self.std = np.std(self.log_returns * np.sqrt(252)).to_numpy()

        self.initial_price = self.stock_data.iloc[-1]

        self.time_delta = 1 / 252
        self.num_steps = int(self.num_of_years / self.time_delta)

        self.normal_rand_nums = np.random.standard_normal((self.num_steps, self.num_of_simulations))
        self.prices = np.exp((self.mean - self.std ** 2 / 2) * self.time_delta + self.std * np.sqrt(self.time_delta) * self.normal_rand_nums)

        self.price_paths = np.zeros((self.num_steps + 1, self.num_of_simulations))
        self.price_paths[0] = self.initial_price

        for t in range(1, self.num_steps + 1):
            self.price_paths[t] = self.price_paths[t - 1] * self.prices[t - 1, :]

        return  self.price_paths

    def __plot__(self):
        plt.rcParams["figure.figsize"] = (12, 6)
        self.fig, ax = plt.subplots()

        ax.plot(self.stock_data.index, self.stock_data, label='Historical Prices', linewidth=2, color='r')
        last_hist_date = self.stock_data.index[-1]
        self.future_dates = pd.date_range(start=last_hist_date, periods=self.num_steps + 1, freq='D')[1:]

        ax.plot(self.future_dates, self.price_paths[1:, :], lw=1, alpha=0.5, color='b')

        ax.set_xlabel('Date')
        ax.set_ylabel('Prices')
        ax.grid(True)
        ax.axvline(x=last_hist_date, color='k', linestyle='--', label='Simulation Prices')
        ax.legend()

        return self.fig

if 'recent_outputs' not in st.session_state:
    st.session_state.recent_outputs = []

with st.sidebar:
    st.title('Stock Price Simulator (GBM)')
    st.write('Created by')
    author = 'Bakytzhan'
    form = f'<a href="{author}" target="_blank" style="text-decoration: none; color: inherit;">`Bakytzhan`</a>'

    st.markdown(form, unsafe_allow_html=True)

    ticker = st.text_input('Enter a stock ticker')
    num_of_days = st.slider('Enter a number of days to fit GBM', min_value=10, max_value=10000, step=1)
    num_of_years = st.number_input('Enter a number of years to simulate', value=1.0, step=0.25)
    num_of_simulations = st.slider('Enter a number of simulations', min_value=50, max_value=1000, step=50)
    calculate_button = st.button('Simulate Prices')

def simulate_prices(ticker, num_of_days, num_of_years, num_of_simulations):

    stock = Stock(ticker, num_of_days, num_of_years, num_of_simulations)
    stock.__data__()
    sim_prices = stock.__price__()
    st.info(f'Historical and Simulated {stock.ticker} Prices')
    stock_price_chart = stock.__plot__()
    st.pyplot(stock_price_chart)

    col1 = st.columns(1)
    st.info('Input summary')
    st.write(f'Ticker: {ticker}')
    st.write(f'Historical number of days : {num_of_years}')
    st.write(f'Number of simulations: {num_of_simulations}')

    st.info('Price Simulation Output')
    df_prices = pd.DataFrame(sim_prices[1:], index=stock.future_dates)
    df_prices.index.name = 'Date'
    st.table(df_prices.iloc[:,:6].head())

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv(index_label=False).encode("utf-8")

    csv_to_download = convert_for_download(df_prices)

    st.download_button(
        label="Download all sim prices",
        data=csv_to_download,
        file_name="sim_prices_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )

if 'first_run' not in st.session_state or st.session_state['first_run']:
    st.session_state.first_run = False
    default_ticker = 'NVDA'
    default_num_of_days = 252
    default_num_of_years = 1
    default_num_of_simulations = 50

    # Perform the default calculation
    simulate_prices(default_ticker, default_num_of_days, default_num_of_years, default_num_of_simulations)

if calculate_button:
    simulate_prices(ticker, num_of_days, num_of_years, num_of_simulations)
