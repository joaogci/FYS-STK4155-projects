import numpy as np
import matplotlib.pyplot as plt

"""
    Analyse the results for the comparison of optimization methods
    Params: 
        tol = 10-e6
        iter_max = 1e6
"""

methods = ["newton", "GD", "SGD"]

time = np.loadtxt("time.txt")
eta_vals = time[:, 0]
time = time[:, 1:3]

# read newton
newton = np.loadtxt("newton.txt")
newton_time = newton[0, 0]
newton_epochs = newton[1:, 0]
newton_mse = newton[1:, 1]

# read GD
gd_time = list()
gd_epochs = list()
gd_mse = list()
gd_last_mse = list()
gd_last_epoch = list()

for i in range(len(eta_vals)):
    gd_time.append(time[i, 0])
    
    tmp = np.loadtxt(f"./GD/GD_eta_{i}.txt")
    gd_mse.append(tmp[:, 1])
    gd_last_mse.append(tmp[-1, 1])
    gd_epochs.append(tmp[:, 0])
    gd_last_epoch.append(tmp[-1, 0])

# read SDG
sgd_time = list()
sgd_epochs = list()
sgd_mse = list()
sgd_last_mse = list()
sgd_last_epoch = list()

for i in range(len(eta_vals)):
    sgd_time.append(time[i, 1])
    
    tmp = np.loadtxt(f"./SGD/SGD_eta_{i}.txt")
    sgd_mse.append(tmp[:, 1])
    sgd_last_mse.append(tmp[-1, 1])
    sgd_epochs.append(tmp[:, 0])
    sgd_last_epoch.append(tmp[-1, 0])

print("Files read! ")

# plots
# 1) -> time vs last mse, each point is a eta
#       line for newton

# plt.figure("time vs last mse")

# plt.plot(gd_time, gd_last_mse, label='GD')
# plt.plot(sgd_time, sgd_last_mse, label='SGD')

# plt.legend()

# 2) -> epochs vs last mse, each point is a eta
#       line for newton

# plt.figure("epochs vs last mse")

# plt.plot(np.log10(gd_last_epoch), gd_last_mse, label='GD')
# plt.plot(np.log10(sgd_last_epoch), sgd_last_mse, label='SGD')

# plt.legend()

# 3) -> time per epochs vs last mse, each point is a eta
#       line for newton

# plt.figure("time/epochs vs last mse")

# plt.plot(np.array(gd_time)/np.array(gd_last_epoch), gd_last_mse, label='GD')
# plt.plot(np.array(sgd_time)/np.array(sgd_last_epoch), sgd_last_mse, label='SGD')

# plt.legend()

# 4) -> epochs vs mse, choose 1 or 2 eta for each

plt.figure("epochs vs mse")

plt.plot(np.log10(gd_epochs[2]), np.log10(gd_mse[2]), label=f'GD, eta={eta_vals[2]}')
plt.plot(np.log10(gd_epochs[4]), np.log10(gd_mse[4]), label=f'GD, eta={eta_vals[4]}')

plt.plot(np.log10(sgd_epochs[2]), np.log10(sgd_mse[2]), label=f'SGD, eta={eta_vals[2]}')
plt.plot(np.log10(sgd_epochs[4]), np.log10(sgd_mse[4]), label=f'SGD, eta={eta_vals[4]}')

plt.plot(np.log10(newton_epochs), np.log10(newton_mse), label='newton')

plt.legend()
plt.show()

