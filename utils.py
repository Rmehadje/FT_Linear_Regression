import csv
import sys
from pathlib import Path
import math

CWD = Path.cwd()
THETAS_PATH = CWD / "thetas.txt"
MINMAX_PATH = CWD / "minmax.txt"

def load_thetas(filepath=THETAS_PATH):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        theta0 = float(lines[0].strip())
        theta1 = float(lines[1].strip())
        return theta0, theta1
    except FileNotFoundError:
        print("Model not trained yet. Please run the training program first.")
        sys.exit(1)
    except (IndexError, ValueError):
        print("Thetas file is corrupted or in an unexpected format.")
        sys.exit(1)

def load_minmax(filepath=MINMAX_PATH):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        min_mileage = float(lines[0].strip())
        max_mileage = float(lines[1].strip())
        if len(lines) >= 4:
            min_price = float(lines[2].strip())
            max_price = float(lines[3].strip())
            return min_mileage, max_mileage, min_price, max_price
        return min_mileage, max_mileage
    except FileNotFoundError:
        print("Min/max file not found. Please run training first.")
        sys.exit(1)
    except (IndexError, ValueError):
        print("Min/max file is corrupted or in unexpected format.")
        sys.exit(1)

def load_data(filename):
    mileages = []
    prices = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                mileage = float(row[0])
                price = float(row[1])
                mileages.append(mileage)
                prices.append(price)
            except ValueError:
                continue
    return mileages, prices

def normalize_data(mileages, prices, min_mileage, max_mileage, min_price, max_price):
    scaled_data = []
    for m, p in zip(mileages, prices):
        scaled_m = (m - min_mileage) / (max_mileage - min_mileage)
        scaled_p = (p - min_price) / (max_price - min_price)
        scaled_data.append((scaled_m, scaled_p))
    return scaled_data

def calculate_rmse(data, theta0, theta1):
    m = len(data)
    if m == 0:
        print("Empty dataset!")
        sys.exit(1)
    mse = sum((theta0 + theta1 * x - y) ** 2 for x, y in data) / m
    return math.sqrt(mse)
