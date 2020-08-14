import csv

from pandas import *
import numpy as np

def save_excel(file_name = "run_results_1.npy"):

    #this serves multiple tests
    sum_ch1 = np.load(file_name)
    channels = 35 #preset channel size
    title = []
    title.append("Run 1")
    total_sum = [item for item in sum_ch1]
    for i in range(len(sum_ch1)):
        for i in range(len(total_sum), channels):
            total_sum.append(0)

    with open("test.csv", 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(total_sum)