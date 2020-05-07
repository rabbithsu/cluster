from sklearn.decomposition import FastICA
import pandas

data_set = pandas.read_csv("0504_behavior.csv")
features = data_set.drop("User_ID", axis = 1)
print features

transformer = FastICA(n_components=3, random_state=0)
X_transformed = transformer.fit_transform(features)
X_transformed.shape
print transformer.predict(features)
