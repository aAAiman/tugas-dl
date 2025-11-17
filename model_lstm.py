import yfinance as yf
import numpy as np

def get_last_price(stock_code):
    stock = yf.Ticker(stock_code + ".JK")
    hist = stock.history(period="5d")  # ambil lebih banyak hari

    if hist.empty:
        return None

    last = hist["Close"].dropna()

    if last.empty:
        return None

    return float(last.iloc[-1])

def predict_stock(stock_code, days):
    last_price = get_last_price(stock_code)

    if last_price is None:
        return ["Data saham tidak ditemukan"]

    hasil = []

    for _ in range(days):
        # perubahan harga realistis: naik/turun 1.5%
        change = last_price * np.random.uniform(-0.015, 0.015)
        next_price = last_price + change
        hasil.append(round(next_price, 2))
        last_price = next_price

    return hasil
