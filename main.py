import pickle
import pandas as pd


# load csv file

df = pd.read_csv("diabetes.csv")

# selection dependant and independant
# Spliting the data
from sklearn.model_selection import train_test_split

X = df.drop("outcome", axis=1)
y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Build a model (Random forest classifier)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
## Evaluating the model

# intentiate the model

classifier = RandomForestClassifier()

# fit the model

classifier.fit(X_train, y_train)

# make pickle file for the model

model = pickle.dump(classifier, open("model.pkl", "wb"))

#   Training part is complete
from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)

@app.route("/")
def home():
    return ("Hello world")
model = pickle.load(open("model.pkl", "rb"))



@app.route("/predict", methods=["POST"])
def predict():
    pregnancies = request.form.get('pregnancies')
    glucose = request.form.get('glucose')
    bloodpressure = request.form.get('bloodpressure')
    skinthickness = request.form.get('skinthickness')
    insulin = request.form.get('insulin')
    bmi = request.form.get('bmi')
    diabetesPedigreeFunction = request.form.get('diabetesPedigreeFunction')
    age = request.form.get('age')
    input_query = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi,
                             diabetesPedigreeFunction, age]])

    result = model.predict(input_query)[0]

    return jsonify({'Having Diabetes': str(result)})
    if __name__ == '__main__':
        app.run(debug=True)

    print(input_query)


    result = model.predict(sc.transform(input_query))
    print(result)
    return jsonify({'Having Diabetes': str(result)})


if __name__ == '__main__':
    app.run(debug=True)

