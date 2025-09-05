from utils import load_thetas, load_minmax

def main():
    theta0, theta1 = load_thetas()
    min_mileage, max_mileage = load_minmax()[:2]

    try:
        try:
            mileage_input = input("Enter mileage: ").strip()
        except KeyboardInterrupt:
            print("\nexiting...")
            return
        mileage = float(mileage_input)
        if mileage < 0:
            print("Mileage cannot be negative. Please enter a positive number.")
            return
    except ValueError:
        print(f"Invalid input '{mileage_input}'! Please enter a valid number.")
        return

    if mileage > 400000:
        print('ERROR: Can not trust model result, aborting.')
        exit()

    if mileage < min_mileage or mileage > max_mileage:
        print(f"WARNING: Mileage is outside the training range. Prediction will be less accurate.")

    price = theta0 + theta1 * mileage
    print(f"Estimated price: {price:.2f}")

if __name__ == "__main__":
    main()
