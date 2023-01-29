#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num_points = 100
plt.rcParams.update({"text.usetex": True})
X = np.random.uniform(size=num_points)
x_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
y_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#%% [markdown]
# # Linear and Noisey
# We'll start by generating a noisey linear relationship. we will generate a
# bunch of x values uniformly on [0, 1) and add gaussian noise to get y

#%%
spread = 0.08
slope = 0.5
X_linear = np.random.normal(0.5, 0.2, size=num_points)
Y_linear = 0.5 * X_linear + np.random.normal(0, spread, size=X.shape) + 0.25

df = pd.DataFrame({"X": X, "Y": Y_linear})

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.scatter(X_linear, Y_linear, marker='x', alpha=0.7)
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlim((-0.04, 1.04))
ax.set_ylim((-0.04, 1.04))
plt.tight_layout()
plt.savefig("linear.png")
plt.show()
# %% [markdown]
# # a circle
#%%
def noisey_circle(x_vals, radius, noise):
    angles = x_vals * 2 * np.pi
    x_circle = np.cos(angles) * radius + 0.5 + np.random.normal(0, noise, size=angles.shape)
    y_circle = np.sin(angles) * radius + 0.5 + np.random.normal(0, noise, size=angles.shape)

    return (x_circle, y_circle)

X_circle, Y_circle = noisey_circle(np.random.uniform(size=num_points),
                                   0.35, 0.02)

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.scatter(X_circle, Y_circle, marker='x', alpha=0.7, c='C1')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlim((-0.04, 1.04))
ax.set_ylim((-0.04, 1.04))
plt.tight_layout()
plt.savefig("circle.png")
plt.show()
# %% [markdown]
# gaussian noise
#%%
X_gauss = np.random.normal(0.5, 0.2, size=num_points)
Y_gauss = np.random.normal(0.5, 0.2, size=num_points)

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.scatter(X_gauss, Y_gauss, marker='x', alpha=0.7, c='C2')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlim((-0.04, 1.04))
ax.set_ylim((-0.04, 1.04))
plt.tight_layout()
plt.savefig("gaussian.png")
plt.show()
# %% [markdown]
# # non-monotonic
# %%
X_nm = np.random.uniform(size=num_points)
Y_nm = (1.5 * X_nm - 1.5 * X_nm**3) + np.random.normal(0, 0.15, size=X_nm.shape) + 0.25

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.scatter(X_nm, Y_nm, marker='x', alpha=0.7, c='C3')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlim((-0.04, 1.04))
ax.set_ylim((-0.04, 1.04))
plt.tight_layout()
plt.savefig("nonmonotonic.png")
plt.show()# %%

# %%
