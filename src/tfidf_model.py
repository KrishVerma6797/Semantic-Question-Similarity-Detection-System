from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TfidfBaseline:

    def __init__(self):
        self.vectorizer=TfidfVectorizer(max_features=5000)
        self.model=LogisticRegression(max_iter=1000)
        
    def train(self,x_train,y_train):
        x_train_vec=self.vectorizer.fit_transform(x_train)
        self.model.fit(x_train_vec,y_train)

    def predict(self,x_test):
        x_test_vec=self.vectorizer.transform(x_test)
        return self.model.predict(x_test_vec)
        
