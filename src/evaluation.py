from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def evaluate(y_true, y_pred, name):

    print("\n", name)
    print("-------------------")

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred))

    return acc
