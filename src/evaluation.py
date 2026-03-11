from sklearn.metrics import accuracy_score, classification_report


def evaluate(y_true, y_pred, name):

    print("\n", name)
    print("-------------------")

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred))

    return acc