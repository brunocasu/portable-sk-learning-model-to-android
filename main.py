from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from pandas import read_csv
from numpy import ravel
import json
import joblib
import pandas as pd
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
import numpy
import numpy as np
import matplotlib.pyplot as plt


## REVERSE COLLECTION
#inv_data = pd.read_csv('data/individual-dataset/inverted-scooter1.csv')
#print(inv_data)
#label_uuid = inv_data.iloc[:,0:2]
#xinv = inv_data.iloc[:,2:len(inv_data.columns)-256]
#yinv = inv_data.iloc[:,2+128:len(inv_data.columns)-128]
#zinv = inv_data.iloc[:,2+256:len(inv_data.columns)]
#fixx = xinv[xinv.columns[::-1]]
#fixy = yinv[yinv.columns[::-1]]
#fixz = zinv[zinv.columns[::-1]]
#fixed_data = pd.concat([label_uuid, fixx, fixy, fixz], axis=1)
#fixed_data.to_csv('data/individual-dataset/fixed-scooter1.csv', index=False)
#exit()

## INGESTION
# combine .csv Datasets
data0 = pd.read_csv('data/individual-dataset/bike0.csv')
data1 = pd.read_csv('data/individual-dataset/scooter1.csv')
data2 = pd.read_csv('data/individual-dataset/walk2.csv')
data3 = pd.read_csv('data/individual-dataset/run3.csv')
data4 = pd.read_csv('data/individual-dataset/bus4.csv')
print("RAW INDIVIDUAL", data0.shape, data1.shape, data2.shape, data3.shape, data4.shape)
mean = 1
std = 0.3
num_samples = 384

### DATA AUGMENTATION
d0 = data0.iloc[:,2:len(data0.columns)]
l0 = data0.iloc[:,0:2]
aug0 = d0.to_numpy()
for r in range(2): ### BIKE DATA - INITAL SIZE = 705 Lines
    aug0 = d0.to_numpy()
    n_rows, n_col = aug0.shape
    for n in range(n_rows):
        samples = numpy.random.normal(mean, std, size=num_samples)
        aug0[n] = np.multiply(aug0[n], samples)
    pd_aug0 = pd.DataFrame(aug0, columns=d0.columns)
    n0 = pd.concat([l0, pd_aug0], axis=1)
    data0 = pd.concat([data0, n0])
### WALK DATA - FINAL SIZE = 2073 Lines

d1 = data1.iloc[0:570,2:len(data1.columns)]
l1 =  data1.iloc[0:570,0:2]
aug1 = d1.to_numpy()
for r in range(2): ### SCOOTER DATA - INITAL SIZE = 883 Lines
    aug1 = d1.to_numpy()
    n_rows, n_col = aug1.shape
    for n in range(n_rows):
        samples = numpy.random.normal(mean, std, size=num_samples)
        aug1[n] = np.multiply(aug1[n], samples)
    pd_aug1 = pd.DataFrame(aug1, columns=d1.columns)
    n1 = pd.concat([l1, pd_aug1], axis=1)
    data1 = pd.concat([data1, n1])
### SCOOTER DATA - FINAL SIZE = 2023 Lines

d2 = data2.iloc[:,2:len(data2.columns)]
l2 =  data2.iloc[:,0:2]
aug2 = d2.to_numpy()
for r in range(2): ### WALK DATA - INITAL SIZE = 691 Lines
    aug2 = d2.to_numpy()
    n_rows, n_col = aug2.shape
    for n in range(n_rows):
        samples = numpy.random.normal(mean, std, size=num_samples)
        aug2[n] = np.multiply(aug2[n], samples)
    pd_aug2 = pd.DataFrame(aug2, columns=d2.columns)
    n2 = pd.concat([l2, pd_aug2], axis=1)
    data2 = pd.concat([data2, n2])
### WALK DATA - FINAL SIZE = 2073 Lines

d3 = data3.iloc[:,2:len(data3.columns)]
l3 =  data3.iloc[:,0:2]
aug3 = d3.to_numpy()
for r in range(1): ### RUN DATA - INITAL SIZE = 1040 Lines
    aug3 = d3.to_numpy()
    n_rows, n_col = aug3.shape
    for n in range(n_rows):
        samples = numpy.random.normal(mean, std, size=num_samples)
        aug3[n] = np.multiply(aug3[n], samples)
    pd_aug3 = pd.DataFrame(aug3, columns=d3.columns)
    n3 = pd.concat([l3, pd_aug3], axis=1)
    data3 = pd.concat([data3, n3])
### RUN DATA - FINAL SIZE = 2080 Lines


d4 = data4.iloc[:,2:len(data4.columns)]
l4 =  data4.iloc[:,0:2]
aug4 = d4.to_numpy()
for r in range(5): ### BUS DATA - INITAL SIZE = 364 Lines
    aug4 = d4.to_numpy()
    n_rows, n_col = aug4.shape
    for n in range(n_rows):
        samples = numpy.random.normal(mean, std, size=num_samples)
        aug4[n] = np.multiply(aug4[n], samples)
    pd_aug4 = pd.DataFrame(aug4, columns=d4.columns)
    n4 = pd.concat([l4, pd_aug4], axis=1)
    data4 = pd.concat([data4, n4])
### BUS DATA - FINAL SIZE = 2184 Lines
#noise = data4
#noise.to_csv('data/noise.csv', index=False)

### COMBINE AXIS DATA
print("AUGMENTED INDIVIDUAL", data0.shape, data1.shape, data2.shape, data3.shape, data4.shape)
data = pd.concat([data0, data1, data2, data3, data4])
lb_uuid = data.iloc[:, 0:2]
x_d = data.iloc[:, 2:len(data.columns)-256]
y_d = data.iloc[:, 2+128:len(data.columns)-128]
y_d.columns = x_d.columns
z_d = data.iloc[:, 2+256:len(data.columns)]
z_d.columns = x_d.columns
x_d.add(y_d)
x_d.add(z_d)
data_combined = pd.concat([lb_uuid, x_d], axis=1)
data_combined.to_csv('data/dataset_combined.csv', index=False)
data_c = pd.read_csv('data/dataset_combined.csv')
print("COMBINED DATASET", data_c.shape)
## REMOVE WRONG SAMPLES
# discard samples with low values
n = 0
for x in range(data_c.shape[0]):
    for y in range(data_c.shape[1] - 2):
        if data_c.iat[x + n, y+2] > 0.05:  #### FILTER VALUE: gain = 0.05
            break
        if y+2 == data_c.shape[1] - 1:
            data_c.drop(x, inplace=True)
            n -= 1

print("FILTERED DATASET", data_c.shape)
### NORMALIZATION
fd = data_c.iloc[:,2:len(data_combined.columns)]
ld = data_c.iloc[:,0:2]
num_fd = fd.to_numpy()
num_ld = ld.to_numpy()
n_rows, n_col = num_fd.shape
#plt.plot(num_fd[7])
#plt.show()
for l in range(n_rows):
    mx = np.amax(num_fd[l])
    num_fd[l] = np.true_divide(num_fd[l], mx)
#plt.plot(num_fd[7])
#plt.show()
pd_fd = pd.DataFrame(num_fd, columns=fd.columns)
pd_ld = pd.DataFrame(num_ld, columns=ld.columns)
data_final = pd.concat([pd_ld, pd_fd], axis=1)

print("NORMALIZED DATASET\n", data_final)
data_final.to_csv('data/dataset_final.csv', index=False)

ml_data = pd.read_csv('data/dataset_final.csv')

## SEGREGATION
training_data, testing_data, training_labels, testing_labels = train_test_split(ml_data.iloc[:, 2:len(data_final.columns)], ml_data.iloc[:, 0])  # return DataFrame obj (pandas)
print("TESTING SET", testing_data.shape,"TRAINING SET", training_data.shape)

### DEVELOPMENT
mlp = MLPClassifier(random_state=1234)
parameters = {'max_iter': (100, 120)}
gs = GridSearchCV(mlp, parameters)
gs.fit(training_data, ravel(training_labels))


## EXECUTE THE CLASSIFIER
print('TRAIN DATA\n', training_data)
print('TEST DATA\n', testing_data)
inference = gs.predict(testing_data)
score = accuracy_score(ravel(testing_labels), inference)
print('deploy score:', score)
print('inference\n', inference)
#
joblib.dump(gs, 'fitted_model.sav')

## CONVERT MODEL TO .pmml
pipeline = PMMLPipeline([("classifier", MLPClassifier(random_state=1122, max_iter=100))])
pipeline.fit(training_data, ravel(training_labels))
sklearn2pmml(pipeline, "model.pmml")
inferencep = pipeline.predict(testing_data)
scorep = accuracy_score(ravel(testing_labels), inferencep)
print('TEST Labels\n', ravel(testing_labels), type(ravel(testing_labels)))
pdlbs = pd.DataFrame(ravel(testing_labels))
pdlbs.to_csv('data/test_labels.csv')
print('PMML inference\n', inferencep, type(inferencep))
inferencep = pd.DataFrame(inferencep)
inferencep.to_csv('data/test_inference.csv')
print('deploy score:', scorep)

## SERIALIZE PMML (DONE IN JAVA)


## TEST WITH REAL TIME DATA
data = pd.read_csv('data/test.csv')
lb_uuid = data.iloc[:, 0:2]
x_d = data.iloc[:, 2:len(data.columns)-256]
y_d = data.iloc[:, 2+128:len(data.columns)-128]
y_d.columns = x_d.columns
z_d = data.iloc[:, 2+256:len(data.columns)]
z_d.columns = x_d.columns
x_d.add(y_d)
x_d.add(z_d)
combined_test = pd.concat([lb_uuid, x_d], axis=1)

n = 0
for x in range(combined_test.shape[0]):
    for y in range(combined_test.shape[1] - 2):
        if combined_test.iat[x + n, y+2] > 0.05:  #### FILTER VALUE: gain = 0.05
            break
        if y+2 == combined_test.shape[1] - 1:
            combined_test.drop(x, inplace=True)
            n -= 1

fd = combined_test.iloc[:,2:len(combined_test.columns)]
ld =  combined_test.iloc[:,0:2]
num_fd = fd.to_numpy()
n_rows, n_col = num_fd.shape

for l in range(n_rows):
    mx = np.amax(num_fd[l])
    num_fd[l] = np.true_divide(num_fd[l], mx)

pd_fd = pd.DataFrame(num_fd, columns=fd.columns)
test_data = pd.concat([ld, pd_fd], axis=1)
rt_label = test_data.iloc[:, 0]
rt_data = test_data.iloc[:, 2:len(test_data.columns)]
inf = pipeline.predict(rt_data)
sc = accuracy_score(ravel(rt_label), inf)
print('RT TEST\nLabels\n', ravel(rt_label))
print('PMML inference\n', inf)
print('deploy score:', sc)
