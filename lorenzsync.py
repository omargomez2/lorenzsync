#-------------------------------
# Working example of chaos synchronization using Lorenz attractor
# osgg.net
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
coupling_strength = 1.5  # Increased coupling strength for better synchronization

# Define the Lorenz system (master)
def lorenz_master(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Define the Lorenz system (slave) with synchronization
def lorenz_slave(state, t, master_signal):
    x, y, z = state
    # Synchronize x by forcing it with the master's x at the current time step, with a stronger coupling
    dxdt = sigma * (master_signal - x) * coupling_strength
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Random initial conditions for both master and slave systems
initial_master = np.random.uniform(-20, 20, size=3)  # Random values between -20 and 20 for the master system
initial_slave = np.random.uniform(-20, 20, size=3)   # Random values between -20 and 20 for the slave system

# Time points for integration
t = np.linspace(0, 50, 10000)

# Solve the Lorenz system (master)
master_solution = odeint(lorenz_master, initial_master, t)

# Solve the Lorenz system (slave) using the master's x value for synchronization
slave_solution = np.zeros_like(master_solution)
slave_solution[0] = initial_slave

# Iterate over the time steps and synchronize the slave system
for i in range(1, len(t)):
    slave_solution[i] = odeint(lorenz_slave, slave_solution[i-1], [t[i-1], t[i]], args=(master_solution[i, 0],))[-1]

# Plot the synchronization of the two systems with thinner lines
plt.figure(figsize=(10, 6))

# Plot the x component of both master and slave
plt.subplot(3, 1, 1)
plt.plot(t, master_solution[:, 0], label='Master x', color='b', linewidth=0.5)
plt.plot(t, slave_solution[:, 0], label='Slave x (synchronized)', color='r', linestyle='--', linewidth=0.5)
plt.title('Chaotic Synchronization of Lorenz Attractors (Random Initial Conditions)')
plt.ylabel('x')
plt.legend()

# Plot the y component of both master and slave
plt.subplot(3, 1, 2)
plt.plot(t, master_solution[:, 1], label='Master y', color='b', linewidth=0.5)
plt.plot(t, slave_solution[:, 1], label='Slave y', color='r', linestyle='--', linewidth=0.5)
plt.ylabel('y')
plt.legend()

# Plot the z component of both master and slave
plt.subplot(3, 1, 3)
plt.plot(t, master_solution[:, 2], label='Master z', color='b', linewidth=0.5)
plt.plot(t, slave_solution[:, 2], label='Slave z', color='r', linestyle='--', linewidth=0.5)
plt.ylabel('z')
plt.xlabel('Time')
plt.legend()

plt.tight_layout()
plt.show()
