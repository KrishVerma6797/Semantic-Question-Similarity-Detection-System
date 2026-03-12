from src.tfidf_model import TfidfBaseline
from src.transformer_model import TransformerSimilarity
from src.evaluation import evaluate
from src.preprocessing import load_data
print('\nLoading Dataset')
x_train,x_test,y_train,y_test,q1_train,q1_test,q2_train,q2_test=load_data("data/questions.csv")
q1_list = X_test_q1.tolist()


## TF-IDF TIMELINE
print("\nTaining TF-IDF baseline")
tfidf_model=TfidfBaseline()
tfidf_model.train(x_train,y_train)
tfidf_pred=tfidf_model.predict(x_test)
tfidf_acc=evaluate(y_test,tfidf_pred,'TF-IDF + Logistic Regression')


## Transformer Model

print("\nRunning transformer similarity model...")
transformer_model=TransformerSimilarity()
bert_pred=transformer_model.predict(q1_test.tolist(),q2_test.tolist())
bert_acc=evaluate(y_test,bert_pred,'Sentence-BERT Similarity')

# MODEL COMPARISON

print("\nModel Comparison")
print("------------------")
print("TF-IDF Accuracy:", tfidf_acc)
print("Sentence-BERT Accuracy:", bert_acc)


# ----------------------------
# Semantic Search Feature
# 
from src.semantic_search import SemanticSearch
search_engine = SemanticSearch(q1_list)
query = input("Enter a question to search: ")
results = search_engine.search(query)
print("\nMost similar questions:")
for r in results:
    print("-", r)

## Interactive Test

while True:
    q1=input("\nEnter question 1:")
    q2=input("Enter question 2:")
    sim = transformer_model.similarity(q1, q2)
    print("Similarity score",sim)
    if sim>0.7:
        print("Duplicate question")
    else:
        print("Different question")


from src.evaluation import plot_confusion_matrix

plot_confusion_matrix(y_test, tfidf_pred, "TF-IDF Confusion Matrix")
plot_confusion_matrix(y_test, transformer_pred, "Sentence-BERT Confusion Matrix")
