import sys
import pandas as pd
import numpy as np

input_data = sys.argv[1]
output_filename = sys.argv[2]

df = pd.read_csv(input_data)
# TODO: Delete null rows
df.dropna()
arr = np.array(df)

print(len(arr[0]))

# TODO: Categorical to Numerical
str_list = []

for i in range(len(arr)):
    for j in range(len(arr[0])):
        if type(arr[i][j]) is str:
            str_list.append(arr[i][j])

str_list = list(set(str_list))
#print(str_list)

main_dict = {}
for i in range(len(str_list)):
    main_dict.update({str_list[i]:float(i/len(str_list))})

for k, v in main_dict.items():
    print(k, ":", v)

for i in range(len(arr)):
    for j in range(len(arr[0])):
        if type(arr[i][j]) is str:
            val = main_dict[arr[i][j]]
            arr[i][j] = val

df = pd.DataFrame(arr)

#print(df1)

# TODO: Scaling

print(arr.shape)
print(df.mean()[2])
print(df.std()[2])


for i in range(len(arr)):
    for j in range(len(arr[0])):
        print(arr[i][j])
        arr[i][j] = (arr[i][j] - df.mean()[j]) / df.std()[j]
        break


df = pd.DataFrame(arr)
print(df)

df.to_csv(r''+output_filename, header=None, index=None)