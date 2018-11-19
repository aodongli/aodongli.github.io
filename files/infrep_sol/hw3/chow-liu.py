import numpy as np

DATA_FOLDER = './data/'
NUM_INSTANCE = 4367
NUM_OBJ = 111
NUM_STATE = 2

data_file = open(DATA_FOLDER + 'chowliu-input.txt', 'rb')

D = np.zeros((NUM_INSTANCE, NUM_OBJ)) # data matrix
for i, line in enumerate(data_file.readlines()):
    line = line.replace('\n', '')
    line = line.split()
    assert len(line) == NUM_OBJ
    for j, w in enumerate(line):
        D[i][j] = float(w)

data_file.close()

# print(A)
print(np.shape(D))

P_single = np.zeros((NUM_OBJ, NUM_STATE))
P_single[:, 1] = np.sum(D, axis=0) / NUM_INSTANCE
P_single[:, 0] = 1 - P_single[:, 1]

print(P_single[0])
print(P_single[2])
def mutual_info(joint_dist, u, v):
    res = 0.0
    n = len(joint_dist)
    for i in range(n):
        for j in range(n):
            if joint_dist[i][j] > 0.0:
                res += joint_dist[i][j] * (np.log(joint_dist[i][j]) - 
                        np.log(P_single[u][i]) - np.log(P_single[v][j]))
    return res

A = np.zeros((NUM_OBJ, NUM_OBJ)) # adjacent matrix
for i in range(NUM_OBJ):
    for j in range(NUM_OBJ - i - 1):
        k = j + i + 1

        # process A[i][k]      
        cnt = np.zeros((NUM_STATE, NUM_STATE))
        for sample in range(NUM_INSTANCE):
            cnt[int(D[sample][i])][int(D[sample][k])] += 1

        assert np.sum(cnt) == NUM_INSTANCE
        
        cnt /= NUM_INSTANCE

        A[k][i] = A[i][k] = mutual_info(cnt, i, k)


# print(P_single)

