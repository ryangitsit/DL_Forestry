
import numpy as np
import os
import matplotlib.pyplot as plt



# assign directory
directory = 'model_training_history\speed'
 
# iterate over files in
# that directory
plt.figure(figsize=(24, 16))
big_list=[]
names=[]
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    #checking if it is a file
    if os.path.isfile(file):
        names.append(str(file)[38:-28])

    list=[]
    with open(file, 'rb') as f:
        for line in f:
            list.append(str(line))

    big_list.append(int(list[1][2:8]))

# print(names)

# print(big_list)
x = names #np.arange(0,len(os.listdir(directory)),1)
y = big_list

# print(len(x),len(y))
# fig, ax = plt.subplots()

# ax.stem(x, y)
# plt.xticks(rotation=60, ha="right")

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.figure(figsize=(28,8))
plt.bar(x, height=y, width=0.15, bottom=None, align='center')
plt.legend()
plt.title("Training Time")
plt.ylabel("Time (ms)")
plt.xlabel("Configuration")
plt.xticks(rotation=60, ha="right")

plt.show()