from sklearn.ensemble import RandomForestClassifier
import numpy as np

class PointwiseLTRModel:
    def __init__(self, n_estimators=100, max_depth=10):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

    def fit(self, X_train_features, y_train_labels):
        """
        X_train_features: list of feature lists [bm25_score, tfidf_score, doc2vec_score, q_len, c_len, ... ]
        y_train_labels: 1 if positive pair, 0 if negative pair
        """
        self.clf.fit(X_train_features, y_train_labels)

    def predict_proba(self, X_test_features):
        """ Returns probability of class 1 """
        if len(self.clf.classes_) == 1:
            return np.zeros(len(X_test_features))
        return self.clf.predict_proba(X_test_features)[:, 1]
