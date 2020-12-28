import numpy as np
B = np.array([0.1,0.05,0.2,0.1]).reshape(2,2)
X = np.array([100.0,200.0])
Y = np.array([3.0,9.0])

x = np.array([100])
y = np.array(3)
b = np.array([[0.1]])

N = 300
B = B/N
while True:
    dX = np.sum(-B * (X.reshape(2, 1).dot(Y.reshape(1, 2))), axis=1)
    X += dX
    Y -= dX
    print(Y)
  #  print(np.sum(-B * (X.reshape(2, 1).dot(Y.reshape(1, 2))), axis=1))
# D = np.array([[2,2,2,3],[6,6,6,7]])
# for i in range(len(D)):
#     cop = np.copy(D[i])
#     cop.shape = (4, 1)
#     # print(cop)
#     bet = B[i]
#     bet = bet * Y
#     infected = np.zeros(4)
#     for j in range(2):
#         infected += D[i]*bet[j]
#     print(infected)
#
# V = np.array([3,4,0])
# vaccines = 3
# profile = V*(vaccines/np.sum(V))
# V = V - profile
# print(V)

print(np.sum(-B*(X.reshape(2, 1).dot(Y.reshape(1, 2))),axis=1))
