import csv
import pandas as pd
import numpy as np

data1 =  np.loadtxt("logi.csv", delimiter=",")
data2 =  np.loadtxt("submission_0.5_0.3_0.2_0.1_666.csv", delimiter=",")
data3 =  np.loadtxt("GradientBoostingClassifier_100.csv", delimiter=",")

hehe12 = 0;
hehe13 = 0;
hehe23 = 0;

for i in range(0,884262):
	sum12 = 0;
	sum13 = 0;
	sum23 = 0;

	for j in range(1,40):
		sum12 += (data1[i][j]-data2[i][j])*(data1[i][j]-data2[i][j])
		sum23 += (data2[i][j]-data3[i][j])*(data2[i][j]-data3[i][j])
		sum13 += (data1[i][j]-data3[i][j])*(data1[i][j]-data3[i][j])

	if (sum12<sum23 and sum12<sum13):
		hehe12+=1;
		for j in range(1,40):
			data1[i][j] = data1[i][j]*0.5+data2[i][j]*0.5

	if (sum23<sum12 and sum23<sum13):
		hehe23+=1;
		for j in range(1,40):
			data1[i][j] = data2[i][j]*0.5+data3[i][j]*0.5

	if (sum13<sum12 and sum13<sum23):
		hehe13+=1;
		for j in range(1,40):
			data1[i][j] = data1[i][j]*0.5+data3[i][j]*0.5

	if (i%10000==0):
		print("gaile");
		print(i);
		print(hehe12);
		print(hehe13);
		print(hehe23);

df = pd.DataFrame(data1)
del df[0]

df.to_csv("loji_666_GR_0.5_555555.csv")


#np.savetxt("xindele.csv", data1, delimiter=",")