import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

def getCognitiveFeatures(X, n_components=25, n_neighbors=10):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    correlation_probabilities = np.zeros_like(distances)
    for i in range(X.shape[0]):
        for j in range(n_neighbors):
            correlation_probabilities[i, j] = 1 / n_neighbors
   
    def nca_objective(W):
        loss = 0
        for i in range(X.shape[0]):
            for j in range(n_neighbors):
                neighbor_index = indices[i, j]
                loss += correlation_probabilities[i, j] * np.dot(X[i] - X[neighbor_index], W.T) @ (X[i] - X[neighbor_index])
        return loss 
    
    # Use PCA to solve the NCA optimization problem
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)
    return pca, transformed_data


dataset = pd.read_csv("Dataset/alzheimers_disease_data.csv")
dataset.fillna(0, inplace = True)#replace missing values
Y = dataset['Diagnosis'].ravel()
dataset.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis = 1,inplace=True)
X = dataset.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle dataset values
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

print(X_train.shape)
features, X_train = getCognitiveFeatures(X_train)
print(X_train.shape)
X_test = features.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier() 
nb = GaussianNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
xgb_cls = XGBClassifier()

estimators = [('knn', knn), ('dt', dt), ('rf', rf), ('nb', nb), ('lr', lr), ('ada', ada), ('xg', xgb_cls)]
vt = VotingClassifier(estimators=estimators)
vt.fit(X_train, y_train)
predict = vt.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)












