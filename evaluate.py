import sys
from utils import load_thetas, load_data, calculate_rmse

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <dataset_path>")
        sys.exit(1)

    theta0, theta1 = load_thetas()
    mileages, prices = load_data(sys.argv[1])

    data = list(zip(mileages, prices))
    rmse = calculate_rmse(data, theta0, theta1)
    print(f"Precision (RMSE) on dataset '{sys.argv[1]}' (price units): {rmse:.2f}")

if __name__ == "__main__":
    main()
