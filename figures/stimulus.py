import matplotlib.pyplot as plt
import numpy as np

try:
    stim = np.genfromtxt('stims/chirp23a.csv')
except:
    stim = np.genfromtxt('../stims/chirp23a.csv')

stim = stim[:15000]

plt.figure(facecolor='white', figsize=(10, 2))

t_axis = np.arange(0, 0.02*len(stim), 0.02)

plt.plot(t_axis, stim)

yrange = plt.ylim()


# ACTIVE RANGE

# dashed vertical lines
plt.plot([5500*.02, 5500*.02], yrange, color='red', linewidth=0.5, linestyle='--')
plt.plot([14500*.02, 14500*.02], yrange, color='red', linewidth=0.5, linestyle='--')

# horizontal line at bottom
# plt.plot([5500*.02, 14500*.02], [yrange[0]*.98, yrange[0]*.98], color='red')

# shaded region
# plt.axvspan(5500*.02, 14500*.02, color='red', alpha=0.2)


# HORIZONTAL AXIS

# small marker
plt.axis('off')
line_x, line_y = 40, -1.2
plt.plot([line_x + 0, line_x + 20], [line_y + 0, line_y + 0], color='k')
plt.text(line_x, line_y-0.5, '20 ms', color='k')

# # axis with gridlines
# for spine in ('left', 'right', 'top'):
#     plt.gca().spines[spine].set_visible(False)
# plt.gca().get_yaxis().set_visible(False)
# plt.gca().get_xaxis().grid(True)
# plt.xlabel('Time (ms)')




plt.show()
