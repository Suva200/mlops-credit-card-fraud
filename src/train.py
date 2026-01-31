from data import load_data
from model import build_model

def train():
    df = load_data("data/train.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    model = build_model()
    model.fit(X, y)

    print("Training completed")

if __name__ == "__main__":
    train()
