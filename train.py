import sys
import matplotlib.pyplot as plt
from utils import load_data, normalize_data, THETAS_PATH, MINMAX_PATH


def compute_loss(data, theta0, theta1):
    m = len(data)
    total_loss = 0.0
    for mileage, price in data:
        prediction = theta0 + theta1 * mileage
        total_loss += (prediction - price) ** 2
    return total_loss / (2 * m)

def train(filename, learning_rate=0.01, iterations=5000):
    mileages, prices = load_data(filename)
    min_mileage, max_mileage = min(mileages), max(mileages)
    min_price, max_price = min(prices), max(prices)

    data = normalize_data(mileages, prices, min_mileage, max_mileage, min_price, max_price)

    m = len(data)
    theta0, theta1 = 0.0, 0.0
    loss_history = []

    for _ in range(iterations):
        sum_errors_theta0 = 0.0
        sum_errors_theta1 = 0.0

        for mileage, price in data:
            prediction = theta0 + theta1 * mileage
            error = prediction - price
            sum_errors_theta0 += error
            sum_errors_theta1 += error * mileage

        theta0 -= learning_rate * (1/m) * sum_errors_theta0
        theta1 -= learning_rate * (1/m) * sum_errors_theta1

        loss = compute_loss(data, theta0, theta1)
        loss_history.append(loss)

    theta1_orig = theta1 * (max_price - min_price) / (max_mileage - min_mileage)
    theta0_orig = min_price + theta0 * (max_price - min_price) - theta1_orig * min_mileage

    with THETAS_PATH.open('w') as f:
        f.write(f"{theta0_orig}\n{theta1_orig}\n")

    with MINMAX_PATH.open('w') as f:
        f.write(f"{min_mileage}\n{max_mileage}\n{min_price}\n{max_price}\n")

    print(f"\nTraining completed:")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Theta0 (intercept): {theta0_orig}")
    print(f"Theta1 (slope): {theta1_orig}")

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.scatter(mileages, prices, color='blue', label='Data points')
    x_vals = [min_mileage, max_mileage]
    y_vals = [theta0_orig + theta1_orig * x for x in x_vals]
    plt.plot(x_vals, y_vals, color='red', label='Regression line')
    plt.title('Mileage vs Price')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), loss_history, color='green')
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)
    train(sys.argv[1])
