from src.train import train


def main():
    results = train("data/creditcard.csv")
    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
