from sklearn.cluster import AgglomerativeClustering
import pandas

data_set = pandas.read_csv("0504_behavior.csv")
features = data_set.drop("User_ID", axis = 1)

hier = AgglomerativeClustering().fit(features)
#hier = AgglomerativeClustering().fit_predict(features)
print hier.labels_
print len(hier.labels_)
