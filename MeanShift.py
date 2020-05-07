from sklearn.cluster import MeanShift
import pandas

data_set = pandas.read_csv("0504_behavior.csv")
features = data_set.drop("User_ID", axis = 1)

ms = MeanShift().fit(features)

print ms.predict(features)
print len(ms.predict(features))

