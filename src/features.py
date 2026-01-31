from sklearn.preprocessing import StandardScaler

def preprocess(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)