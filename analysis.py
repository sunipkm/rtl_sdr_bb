# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# %%
data = np.fromfile('test.bin', dtype = np.single, count = -1)
# %%
bb_i = data[0:-2:2]
bb_q = data[1:-1:2]
# %%
print(bb_i.shape, bb_q.shape)
# %%
bbdata = bb_i + 1j*bb_q
# %%
cfreq = 437.5e6
samp = 250e3
period = 1/samp
bw = 20e3
sampwindow = int(bw)
# %%
num_frames = int(bbdata.shape[0]//sampwindow)
print(num_frames)
# %%
# Set up FFT
freq = np.fft.fftshift(np.fft.fftfreq(sampwindow)) + cfreq
# %%
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], c = 'k')

def init():
    ax.set_xlim(freq.min(), freq.max())
    return ln,

def update(frame):
    fig.suptitle("Time: %.2f/%.2f s"%((frame + 1)*period*sampwindow, num_frames*period*sampwindow))
    ydata = np.abs(np.fft.fftshift(np.fft.fft(bbdata[frame*sampwindow:(frame+1)*sampwindow], norm="ortho")))
    ax.set_ylim(ydata.min(), ydata.max() * 1.2)
    ln.set_data(freq, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=num_frames,
                    init_func=init, interval = period*sampwindow*1e3, blit=False, repeat = False)
plt.show()
# %%
