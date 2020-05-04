from sklearn.mixture import GaussianMixture
import pandas

data_set = pandas.read_csv("0504_behavior.csv")
features = data_set.drop("User_ID", axis = 1)

gmm = GaussianMixture(n_components=3).fit(features)
X_transformed = gmm.fit(features)
print X_transformed.score_samples(features)
print len(X_transformed.score_samples(features))

