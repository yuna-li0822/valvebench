"""Small MLP baseline using sklearn if available, else raises."""


try:
    from sklearn.neural_network import MLPRegressor
except Exception:
    MLPRegressor = None


class MLPBaseline:
    def __init__(self, **kwargs):
        if MLPRegressor is None:
            raise RuntimeError("scikit-learn required for MLPBaseline")
        self.model = MLPRegressor(**kwargs)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)
