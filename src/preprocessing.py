import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):

    data = pd.read_csv(path)
    data = data.dropna()

    q1 = data["question1"]
    q2 = data["question2"]
    labels = data["is_duplicate"]

    X_train_q1, X_test_q1, X_train_q2, X_test_q2, y_train, y_test = train_test_split(
        q1, q2, labels, test_size=0.2, random_state=42
    )

    return X_train_q1, X_test_q1, X_train_q2, X_test_q2, y_train, y_test
