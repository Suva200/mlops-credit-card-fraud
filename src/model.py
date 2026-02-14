from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelFactory:
    @staticmethod
    def get_all_models(random_state: int = 42):
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "SVM": SVC(
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        }
