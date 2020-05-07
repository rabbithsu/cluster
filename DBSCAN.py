from sklearn.cluster import DBSCAN
import pandas

data_set = pandas.read_csv("0504_behavior.csv")
features = data_set.drop("User_ID", axis = 1)

db = DBSCAN().fit(features)
#db = DBSCAN().fit_predict(features)
#-1 for noisy samples
print db.labels_
print len(db.labels_)
