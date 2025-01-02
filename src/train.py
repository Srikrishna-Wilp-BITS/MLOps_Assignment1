import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                        test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved!")


if __name__ == "__main__":
    train_model()
