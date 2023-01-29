#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif',size=16)
plt.rcParams.update({"text.usetex": True})
import seaborn as sns

dice_set = [4, 6, 8, 10, 12, 20]
num_trials = 50

# plotting constants
ticks = [0., 5., 10., 15., 20.]

def alice_roll(dice: list):
    faces = np.random.choice(dice)
    roll = np.random.choice(faces)

    return (faces, roll + 1)


def bob_guess(hint_type, hint):
    if hint_type == 'none':
        return np.random.choice(max(dice_set)) + 1
    elif hint_type == 'die':
        return np.random.choice(hint)
    elif hint_type == 'parity':
        guesses = [i for i in range(21) if i % 2 == hint]
        return np.random.choice(guesses)
    elif hint_type == 'value':
        return hint

# no hint
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(7, 7))
rolls = np.zeros(num_trials)
guesses = np.zeros(num_trials)

for i in range(num_trials):
    _, roll = alice_roll(dice_set)
    guess = bob_guess('none', 0)

    rolls[i] = roll
    guesses[i] = guess


ax[0, 0].scatter(rolls, guesses, c="C0")
ax[0, 0].set_title("Alice Silent")
ax[0, 0].set_aspect("equal")
ax[0, 0].set_xlabel("Alice's roll")
ax[0, 0].set_ylabel("Bob's guess")

# number of faces hint
rolls = np.zeros(num_trials)
guesses = np.zeros(num_trials)

for i in range(num_trials):
    die, roll = alice_roll(dice_set)
    guess = bob_guess('die', die)

    rolls[i] = roll
    guesses[i] = guess

ax[0, 1].scatter(rolls, guesses, c="C1")
ax[0, 1].set_title("Alice tells which die")
ax[0, 1].set_aspect("equal")
ax[0, 1].set_xlabel("Alice's roll")
ax[0, 1].set_ylabel("Bob's guess")

# parity hint
rolls = np.zeros(num_trials)
guesses = np.zeros(num_trials)

for i in range(num_trials):
    die, roll = alice_roll(dice_set)
    guess = bob_guess('parity', roll % 2)

    rolls[i] = roll
    guesses[i] = guess

ax[1, 0].scatter(rolls, guesses, c="C2")
ax[1, 0].set_title("Alice tells something else")
ax[1, 0].set_aspect("equal")
ax[1, 0].set_xlabel("Alice's roll")
ax[1, 0].set_ylabel("Bob's guess")

# value hint
rolls = np.zeros(num_trials)
guesses = np.zeros(num_trials)

for i in range(num_trials):
    die, roll = alice_roll(dice_set)
    guess = bob_guess('value', roll)

    rolls[i] = roll
    guesses[i] = guess

ax[1, 1].scatter(rolls, guesses, c="C3")
ax[1, 1].set_title("Alice tells rolled value")
ax[1, 1].set_aspect("equal")
ax[1, 1].set_xticks(ticks)
ax[1, 1].set_xlim(-1, 21)
ax[1, 1].set_yticks(ticks)
ax[1, 1].set_ylim(-1, 21)
ax[1, 1].set_xlabel("Alice's roll")
ax[1, 1].set_ylabel("Bob's guess")
plt.tight_layout()
plt.savefig("alice_bob_game.pdf")
plt.show()
# %% [markdown]
# # distributions for parity hint
#%%
fig, ax = plt.subplots(ncols=2, sharey=True)
info_trials = 2000
rolls = np.zeros(info_trials)
guesses = np.zeros(info_trials)

for i in range(info_trials):
    die, roll = alice_roll(dice_set)
    guess = bob_guess('parity', roll % 2)

    rolls[i] = roll
    guesses[i] = guess

roll_vals, roll_counts = np.unique(rolls, return_counts=True)
guess_vals, guess_counts = np.unique(guesses, return_counts=True)

roll_freq = roll_counts / info_trials
guess_freq = guess_counts / info_trials

ax[0].bar(roll_vals, roll_freq, color="C2")
ax[1].bar(guess_vals, guess_freq, color="C2")

ax[0].set_xlabel("Rolled Values")
ax[1].set_xlabel("Guessed Values")
ax[0].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("alice_bob_hists.pdf")
plt.show()
# %%
# we'll add to these values to get the entropy
roll_entropy = 0
guess_entropy = 0

# loop over all of the possible values
for val in range(max(dice_set) + 1):

    # this logic ensures we have a nonzero frequency for the given value
    if val in roll_vals:

        # we use the empirical frequencies as the probabilities
        roll_prob = roll_freq[roll_vals == val][0]

        # update the entropy according to our formula
        roll_entropy += -roll_prob * np.log2(roll_prob)

    # same proceedure for the guesses    
    if val in guess_vals:
        guess_prob = guess_freq[guess_vals == val][0]
        guess_entropy += -guess_prob * np.log2(guess_prob)
    
print(f"The entropy of Alice's rolls is {roll_entropy:.2f} bits")
print(f"The entropy of Bob's guessess is {guess_entropy:.2f} bits")

# %%

# get the joint_distribution
joint_dist = np.zeros((21, 21))

for roll, guess in zip(rolls, guesses):
    roll_i = int(roll)
    guess_i = int(guess)
    joint_dist[roll_i, guess_i] += 1

joint_dist /= info_trials

jhist = plt.imshow(joint_dist.T, cmap="Greens")
cbar = plt.colorbar(jhist)

cbar.set_label("Probability")
plt.xlabel("Rolled Value")
plt.ylabel("Guessed Value")
plt.xticks(ticks)
plt.yticks(ticks)
plt.savefig("alice_bob_joint_hist.pdf")
plt.show()
# %%
# initialize the joint entropy at zero
joint_entropy = 0

# we have two sums in our joint entropy expression
for ri in range(joint_dist.shape[0]):
    for gi in range(joint_dist.shape[1]):

        # again we'll check that our frequencies are nonzero
        if joint_dist[ri, gi] > 0:

            # update the entropy
            joint_entropy += -joint_dist[ri, gi] * np.log2(joint_dist[ri, gi])

print(f"The joint entropy is {joint_entropy:.2f} bits")
# %%
