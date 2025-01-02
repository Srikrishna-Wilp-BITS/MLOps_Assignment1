from src.train import train_model


def test_train_model():
    try:
        train_model()
    except Exception as e:
        assert False, f"Training failed: {e}"

