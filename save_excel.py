from pandas import DataFrame
import numpy as np

def save_excel():
    file_name = "run_results_1.npy"
    sum_ch1 = np.load(file_name)
    total_sum = []
    total_sum.append(sum_ch1)
    check = 0
    title = []
    for j in range(len(total_sum)):
        if check < len(total_sum[j]):
            check = len(total_sum[j])
        title.append("Run 1")

    index = np.arange(0, check, 1)

    for k in range(len(total_sum)):
        if len(total_sum[k]) < check:
            for l in range(len(total_sum[k]), check):
                total_sum[k].append(0)

    TTotal_sum = list(map(list, zip(*total_sum)))
    df = DataFrame(data=TTotal_sum, columns=title)
    df.to_excel("testfile" + ".xlsx", index=False, sheet_name="Results")

save_excel()