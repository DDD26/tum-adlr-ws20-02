
import numpy as np
from Env import Environment
from objective_function import Objective
import matplotlib.pyplot as plt

pos = np.array([[10,10],[40,50],[40,10]])
size = np.array([[10,20],[10,40],[20,30]])
bound = np.array([64,64])
start = np.array([0.1,0.1])
end = np.array([6,6])
opt_num = 20
sp_num = 5
co_num = 20
w = 1


### gradient descent ###
env = Environment(pos, size, bound)
obj = Objective(start, end, opt_num, sp_num, co_num, env, w)
ob_fun = obj.ob_fun
ob_der_fun = obj.ob_der_fun

iter_num = 200
lr = 0.2
x0 = obj.initial()
for i in range(iter_num):
    x0 -= lr*ob_der_fun(x0)
b = env.dis
plt.imshow(b)

x0 = x0*10-0.5
print(x0)
plt.plot(x0.T[1],x0.T[0])
plt.show()
### objective demo ###