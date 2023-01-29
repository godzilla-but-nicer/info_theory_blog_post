#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import entropy

rng = np.random.default_rng(666)

def dartboard_regions(regions, colormap='colorblind'):
    cmap = sns.color_palette(colormap, as_cmap=True)
    wedges = []
    for wi in range(regions):
        start_deg = wi * 360 / regions
        end_deg = (wi+1) * 360 / regions % 360 
        wedges.append(Wedge((0, 0), 1, start_deg, end_deg, color=cmap[wi % 8]))
    return wedges

def throw_darts(ndarts):
    theta = rng.uniform(0, 360, size=ndarts)
    radius = rng.uniform(0, 1, size=ndarts)

    # trig lol
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return(x, y)

def draw_darts(x, y, ax1, marker_size=1000):
    ax1.scatter(x, y, marker='1', c='k', s=marker_size, zorder=10)
    ax1.scatter(x, y, marker='o', c='k', s=60, zorder=10)
    
#%%
# im going to do all the darts here so i can fiddle with the plots w/o moving them
simple_dx, simple_dy = throw_darts(1)
four_dx, four_dy = throw_darts(1)
three_dx, three_dy = throw_darts(1)
eight_dx, eight_dy = throw_darts(1)
uneven_dx, uneven_dy = throw_darts(1)
# %% [markdown]
# # Entropy
# Entropy can be most simply described as the number of yes/no questions we
# need to answer a given question. Let's start with a simple example of a
# dart board.
#
# Say we've thrown a dart at a dart board. We are practiced enough to always
# hit the board, but novice enough not to aim within the board. How many yes/no
# questions do we need to ask in order to determine what color on the board our
# dart has landed on?
#
# ## The Simplest Case
# In the simplest case with only two colors, this way of posing the question
# starts to make sense. We can first ask 'Is the dart in the green region?'
# and whatever the answer is, we've located the dart!
# 
# This means that before we threw the dart we had an uncertainty of where it
# would land equal to one yes/no question. A *binary* question if you will
#%%
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(2)
for br in board:
    ax.add_artist(br)

draw_darts(simple_dx, simple_dy, ax)
# plt.savefig('two_frame1.png')
plt.show()
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(2)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, 0), 2, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)


draw_darts(simple_dx, simple_dy, ax)
# plt.savefig('two_frame2.png')
plt.show()
# %% [markdown]
# ## Adding regions on the board
#
# If we change the number of colors we should expect that we also will change
# the uncertainty about where it will land. If we double the number of colors,
# we will be less sure of the color on which the dart will land.
#%%
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(4)
for br in board:
    ax.add_artist(br)

draw_darts(four_dx, four_dy, ax)
#plt.savefig('frame1.png')
plt.show()

# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# get the frames for the animation
# draw board and darts
board = dartboard_regions(4)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)

draw_darts(four_dx, four_dy, ax)
#plt.savefig('frame2.png')
plt.show()

# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# get the frames for the animation
# draw board and darts
board = dartboard_regions(4)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.add_artist(Rectangle((-1, 0), 1, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)
ax.axvline(0, color='grey', lw=8)

draw_darts(four_dx, four_dy, ax)
#plt.savefig('frame3.png')
plt.show()
# %% [markdown]
# In this case we first ask 'Is the dart in the top half?' and get the answer
# 'yes'. This allows us to *exclude* the bottom half of the dart board. We know
# that the dart can't be there!
# 
# Next, we ask 'is the dart in the left half?' and get the answer 'no' which
# lets us exclude the left half. This tells us that the dart can only be in
# the top right, or green, section.
# 
# The fact that we had to ask two questions tells us that our uncertainty was
# greater about the color of the region the dart landed on. This should make
# sense, your more likly to be wrong guessing 1 in 4 than 1 in 2. As more
# values are added our ability to guess the outcome---our uncertainty---increases.
# %%
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(8)
for br in board:
    ax.add_artist(br)

draw_darts(eight_dx-0.85, eight_dy+0.05, ax)
#plt.savefig('eight_frame1.png')
plt.show()

# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(8)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)

draw_darts(eight_dx-0.85, eight_dy+0.05, ax)
#plt.savefig('eight_frame2.png')
plt.show()
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(8)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.add_artist(Rectangle((0, 0), 1, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)
ax.axvline(0, color='grey', lw=8)

draw_darts(eight_dx-0.85, eight_dy+0.05, ax)
#plt.savefig('eight_frame3.png')
plt.show()

# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(8)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.add_artist(Rectangle((0, 0), 1, 1, color='white', alpha=0.8))
ax.add_artist(Wedge((0, 0), 1, 90, 135, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)
ax.axvline(0, color='grey', lw=8)
ax.axline((-1.1, 1.1), (1.1, -1.1), color='grey', lw=8)

draw_darts(eight_dx-0.85, eight_dy+0.05, ax)
#plt.savefig('eight_frame4.png')
plt.show()

# %%[markdown] 
# Again the uncertainty increased. We had to ask three questions to
# identify the location of the dart on a board of eight colors.Each time we have
# doubled the number of options, the number of questions we've had to ask has
# increased by one. This is no accident, it is a consequence of the fact that we
# are asking binary questions. The number of questions we need to ask is related
# to the base-2 logrithm of the number of options. $$ \log_2 2 = 1 \\
# \log_2 4 = 2 \\
# \log_2 8 = 3 \\
# ... $$
#
# ## Partial yes/no questions?
# 
# In this dart board example, we formulated yes/no questions as straight lines
# that split the space in half through the center of the circle.
# 
# In contrast to the previous examples our third line was not orthogonal. This
# may then apear to be a different case but it is not. With respect to the
# remaining space the third line still divided it perfectly in half. With the
# first two questions we excluded everything but the top left quarter. The third
# line devided that quarter into eigths.
#
# We shall now turn toward a case where we do not split the remaining space
# neatly in half. This is far more common as it occurs any time the number of
# options is not a power of two!
#
# If we have three options we should expect to need between one and two yes/no
# questions. However, we can no longer rely on always dividing the remaining
# space in half as we have done before. Instead after the first cut we are left
# with an imbalance in the colors on the dart board.



# %%
# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(3)
for br in board:
    ax.add_artist(br)

draw_darts(three_dx+0.4, three_dy+0.9, ax)
plt.savefig('three_first_frame1.png')
plt.show()


# initialize the figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# draw board and darts
board = dartboard_regions(3)
for br in board:
    ax.add_artist(br)

ax.add_artist(Rectangle((-1, -1), 2, 1, color='white', alpha=0.8))
ax.axhline(0, color='grey', lw=8)

draw_darts(three_dx+0.4, three_dy+0.9, ax)
plt.savefig('three_first_frame2.png')
plt.show()

#%% [markdown] 
# Unlike before, after our first cut we are not actually guessing
# in the dark. The dart is more likly to be in the green region than the purple
# because its twice as large. On average, we don't need a whole yes/no question
# to locate the dart, sometimes we can just guess.
# 
# We will soon turn our attention to calculating exactly how many questions we
# need on average. First we will need to build up to that calculation.
#%%
# ## Equal likelihood
#
# The fact that we did not need a whole two cuts in the three color dart board
# illustrates something very important: for a given number of outcomes,
# *uncertainty is maximized when all outcomes are equally likely*. We should
# require that this be true for an measure of uncertainty. Its hardest to guess
# about equally likely outcomes!
#

#%%
cmap = sns.color_palette('colorblind', as_cmap=True)
angles = [0, 0.99*360, 0]

ext_dx, ext_dy = throw_darts(10)

for i in range(10):
    wedges = []
    for wi in range(2):
        start_deg = angles[wi]
        end_deg = angles[wi+1] 
        wedges.append(Wedge((0, 0), 1, start_deg, end_deg, color=cmap[wi % 8]))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


    for br in wedges:
        ax.add_artist(br)


    draw_darts(uneven_dx, uneven_dy, ax)
    plt.savefig(f"ext_frame{i}.png")
    plt.show()
#%% [markdown]
# ## Nestedness
# 
# We've framed the introduction above in way that will hopefully highlight
# another key point. Take the four-color dartboard. In reality we simply threw a
# single dart at one of four options, yet the description was of a sequential
# *nested* process in which we looked at the top or bottom half and then the
# left or right. We can visualize this process as such.

# %%
cmap = sns.color_palette('colorblind', as_cmap=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_aspect('equal')
ax.set_ylim((-2, 2))
ax.set_xlim((-0.5, 2.5))
ax.set_xticks([])
ax.set_yticks([])
crad = 0.2

# lines first
ax.plot((0, 1), (0, 1), color='black', lw=2)
ax.plot((0, 1), (0, -1), color='black', lw=2)
ax.plot((1, 2), (1, 1.5), color='black', lw=2)
ax.plot((1, 2), (1, 0.5), color='black', lw=2)
ax.plot((1, 2), (-1, -1.5), color='black', lw=2)
ax.plot((1, 2), (-1, -0.5), color='black', lw=2)


# all of the points for the diagram
root = Circle((0, 0), crad, color='black', zorder=100)
b1 = Circle((1, 1), crad, color='white', ec='black', zorder=99)
b1_upper = Wedge((1, 1), crad, 0, 180, color='grey', ec='black', zorder=100)
b2 = Circle((1, -1), crad, color='white', ec='black', zorder=99)
b2_lower = Wedge((1, -1), crad, 180, 0, color='grey', ec='black', zorder=200)
l1 = Circle((2, 1.5), crad, color=cmap[0], ec='black', zorder=100)
l2 = Circle((2, 0.5), crad, color=cmap[1], ec='black', zorder=100)  
l3 = Circle((2, -0.5), crad, color=cmap[2], ec='black', zorder=100)
l4 = Circle((2, -1.5), crad, color=cmap[3], ec='black', zorder=100)
for a in [root, b1, b1_upper, b2, b2_lower, l1, l2, l3, l4]:
    ax.add_artist(a)

# probability labels
ax.text(0.45, 0.5, s='1/2', ha='center', va='bottom', rotation=45, size='x-large')
ax.text(0.55, -0.5, s='1/2', ha='center', va='bottom', rotation=-45, size='x-large')
ax.text(1.45, 1.25, s='1/2', ha='center', va='bottom', rotation=45/2, size='x-large')
ax.text(1.55, 0.75, s='1/2', ha='center', va='bottom', rotation=-45/2, size='x-large')
ax.text(1.55, -1.25, s='1/2', ha='center', va='bottom', rotation=-45/2, size='x-large')
ax.text(1.45, -0.75, s='1/2', ha='center', va='bottom', rotation=45/2, size='x-large')

plt.show()
# %% [markdown] 
# The diagram shows how we first cut through the board to chose
# the top or bottom half, then asked about the colors. Each was a binary choice
# of two equally likely outcomes. Each branch is labeled with its probability of
# occuring---the probability of the answer to our question of it being "yes."
#
# We can follow the branches by multiplying their probabilities together to see
# that each outcome still occurs with a probability of $\frac{1}{4}$. %%
cmap = sns.color_palette('colorblind', as_cmap=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_aspect('equal')
ax.set_ylim((-2, 2))
ax.set_xlim((-0.5, 2.5))
ax.set_xticks([])
ax.set_yticks([])
crad = 0.2

# lines
ax.plot((0, 2), (0, 1.5), color='black', lw=2)
ax.plot((0, 2), (0, 0.5), color='black', lw=2)
ax.plot((0, 2), (0, -0.5), color='black', lw=2)
ax.plot((0, 2), (0, -1.5), color='black', lw=2)

# dots
root = Circle((0, 0), crad, color='black', zorder=100)
l1 = Circle((2, 1.5), crad, color=cmap[0], ec='black', zorder=100)
l2 = Circle((2, 0.5), crad, color=cmap[1], ec='black', zorder=100)  
l3 = Circle((2, -0.5), crad, color=cmap[2], ec='black', zorder=100)
l4 = Circle((2, -1.5), crad, color=cmap[3], ec='black', zorder=100)
for a in [root, l1, l2, l3, l4]:
    ax.add_artist(a)

# branch labels
ax.text(1, 0.75, s='1/4', ha='center', va='bottom', rotation=45, size='x-large')
ax.text(1, 0.25, s='1/4', ha='center', va='bottom', rotation=45/2, size='x-large')
ax.text(1, -0.25, s='1/4', ha='center', va='bottom', rotation=-45/2, size='x-large')
ax.text(1, -0.75, s='1/4', ha='center', va='bottom', rotation=-45, size='x-large')

plt.show()
# %% [markdown] 
# Given that these two diagrams are equavalent, our measure of
# uncertainty should produce the same result for either of them. The nested
# structure and the intuition of cutting accross the dart board should help
# here.
#
# The uncertainty about whether the dart is in the top or bottom half is there
# regardless of which branching structure we are thinking about. Then there is
# some additional uncertainty about the specific colors within those half. Of
# course, we only need to address each of these additional sources of
# uncertainty half of the time, half of the time we care about uncertainty
# within the top, half of the time we care about uncertainty within the bottom
# for the bottom.
#
# Lets say we want to call our uncertainty measure $H$ for no particular reason.
# We can formalize this logic as follows: 
# $$ H(\{P_{purple}, P_{green},
# P_{yellow}, P_{orange}\}) = H(\{P_{top}, P_{bottom}\}) + P_{top}
# H(\{P_{purple}^{nested}, P_{green}^{nested}\}) + P_{bottom} H(\{P_{yellow}^{nested}, P_{orange}^{nested}\}) $$
#
# To make this concrete with the values shown in the diagrams above this is:
# 
# $$ H(\{\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\}) = 
# H(\{\frac{1}{2}, \frac{1}{2}\}) + \frac{1}{2} H(\{\frac{1}{2}, \frac{1}{2}\}) +
# \frac{1}{2} H(\{\frac{1}{2}, \frac{1}{2}\}) $$
#
# ## Continuity
# 
# Let's say we change our four-color dart board slightly. Say we make the purple
# region slightly larger and the orange region slightly smaller to compensate.
# %%
cmap = sns.color_palette('colorblind', as_cmap=True)
angles = [0, 90, 195, 270, 0]
wedges = []
for wi in range(4):
    start_deg = angles[wi]
    end_deg = angles[wi+1] 
    wedges.append(Wedge((0, 0), 1, start_deg, end_deg, color=cmap[wi % 8]))

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

for br in wedges:
    ax.add_artist(br)

draw_darts(uneven_dx, uneven_dy, ax)
# plt.savefig('uneven_board.png')
plt.show()
# %% [markdown] The board is pretty similar. We've established that the colors
# were all equally likely so we should require fewer than 2 questions. However,
# the regions are all still roughly the same size so the difference also
# shouldn't be very large.
#
# In fact, if we make arbitrarily small changes to the board, we should see
# corresponding abitrarily small changes in our uncertainty measurement.
# Similarly when we make large changes to the board we should see large changes
# in our measurement. This is what we mean by *continuity* of our measurement
# and it is the third and final requirement we need.
#
# ## Axioms and Entropy
#
# Our three requirements follow the logic of Claude Shannon's three axioms. To
# summarize:
#
# 1. **Continuity**: $H(X)$ should produce arbitrarily small values for suitably
#    small changes to $X$
# 1. **Maximum**: For a fixed number of elements, $N$, $H(X)$ should be at its
#    maximum when all probabilities $p$ are equal---$p(x_i) = \frac{1}{N},
#    \forall i$
# 1. **Coarse-Graining**: If we combine outcomes in $p$, then we nest the $H$
#    for the combined choices inside of the overall $H$ weighed by its
#    probability
#
# Amazingly, there is *exactly one* formula for H that satisfies these axioms:
#
# $$ H(X) = -K \sum_{i=1}^N p(x_i) \log p(x_i) $$
#
# Normally we drop the scaling constant $K$ and use the base-two logarithm to
# give us the formula:
#
# $$ H(X) = - \sum_{i=1}^N p(x_i) \log_2 p(x_i) $$
#
# Using the base-two logarithm gives us a value in units of bits---binary
# digits, answers to yes/no questions. We can use this measurement on any
# probability distribution to measure its uncertainty. Of course its trickier on
# continuous probability distributions. Specifically what it tells us is the
# number of yes/no questions we need to ask on average if we are maximally
# efficient in the questions we ask.
#
# Lets finish up by measuring the uncertainty of our dartboards to make sure it
# works how we think it should.
#
# ## Calculating the entropy of some dartboards
#
# We are now equipped with what we need to calculate the entropy of the
# dartboards we've seen above. We can check whether our claims about cutting the
# dartboard in half are consistant and figure out how many parts of a yes/no
# question we need to answer where the dart is on a three-color board.
#
# First, we will implement a function to calculate the entropy.

#%%
import numpy as np
def h(x: np.array) -> float:
    return -np.sum(x * np.log2(x))

#%% [markdown] 
# Now we just need to encode the dartboards as probability
# distributions. The boards we've looked at so far, except the uneven one, are
# quite easy to encode. All of their regions are uniform in size.
#%%
# two region board
x_two = np.array([1/2, 1/2])

# more succinctly
x_three = np.ones(3) * 1/3
x_four = np.ones(4) * 1/4
x_eight = np.ones(8) * 1/8

print(f"Entropy of the two-color dart board: {h(x_two):.3f}")
print(f"Entropy of the four-color dart board: {h(x_four):.3f}")
print(f"Entropy of the eight-color dart board: {h(x_eight):.3f}")
print(f"Entropy of the three-color dart board: {h(x_three):.3f}")
# %%
