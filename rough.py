# // read grid using file or reading directly taken from image

def read_grid(filename = None):
    file = open(filename,"r")
    for line in file:
        pass
    return line

line = read_grid("../puzzles.txt")

line

len(line)

len(digit)

line[0]

digit[0].shape

import pandas as pd

l_digit = digit.copy()

d = []
for i in range(len(l_digit)):
    d.append(np.reshape(l_digit[i],(-1,)))

d = np.array(d)

d.shape

l = list(line)

x = pd.Series(data=l)

x

y = pd.DataFrame(data=d,dtype=np.uint8)

y.shape

y.info()

df = pd.concat([x,y],axis=1)

df.dtypes

z = df.iloc[-1]

z = z[1:]

c = np.array(z)

c = c.reshape(28,28,1)

plt.imshow(c)