# Sklearn processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Sklearn classification algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Sklearn classification model evaluation functions
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Matplotlib for charting
import matplotlib.pyplot as plt

##

# Load built-in sample data set
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Define the X (input) and y (target) features
X = data.data
y = data.target

# Rescale the input features
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(data.data)

# Split into train (2/3) and test (1/3) sets
test_size = 0.2
seed = 189
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Build and fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the training data
predictions = model.predict(X_train)

# Plot the confusion matrix
print(confusion_matrix(y_train, predictions))

# Classification report
print(classification_report(y_train, predictions, target_names=data.target_names))

# Define a function to plot the ROC/AUC
def plotRocAuc(model, X, y):

    probabilities = model.predict_proba(X)
    probabilities = probabilities[:, 1]  # keep probabilities for first class only

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y, probabilities)

    # Plot the "dumb model" line
    plt.plot([0, 1], [0, 1], linestyle='--')

    # Plot the model line
    plt.plot(fpr, tpr, marker='.')
    plt.text(0.75, 0.25, "AUC: " + str(round(roc_auc_score(y, probabilities),2)))

    # show the plot
    plt.show()

##
# ROC / AUC
plotRocAuc(model, X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print(type(model).__name__)
print("----------------------------------")
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=data.target_names))

print("\nROC / AUC:")
plotRocAuc(model, X_test, y_test)


