# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as sg
# %%
data = np.fromfile('20211022_214858_368_49277', dtype = np.single, count = -1)
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
sampwindow = int(25e3)
# %%
num_frames = int(bbdata.shape[0]//sampwindow)
print(num_frames)
# %%
# Set up FFT
freq = np.fft.fftshift(np.fft.fftfreq(sampwindow, d = period)) + cfreq 
binsize = samp/bw
print("FFT Bin Size = %.2f Hz"%(binsize))
# %%
# Set up max hold
maxhold = np.zeros(freq.shape, dtype = float) - np.inf
minhold = np.zeros(freq.shape, dtype = float) + np.inf
# %%
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], c = 'b')
ln2, = plt.plot([], [], c = 'r')

def init():
    ax.set_xlim(cfreq - 0.5 * bw, cfreq + 0.5 * bw)
    return ln,

def update(frame):
    fig.suptitle("Time: %.2f/%.2f s"%((frame + 1)*period*sampwindow, num_frames*period*sampwindow))
    ydata = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(bbdata[frame*sampwindow:(frame+1)*sampwindow], norm="ortho"))))
    maxhold[np.where(ydata > maxhold)] = ydata[np.where(ydata > maxhold)]
    minhold[np.where(ydata < minhold)] = ydata[np.where(ydata < minhold)]
    ax.set_ylim(minhold.min(), maxhold.max() * 1.2)
    ln.set_data(freq, sg.savgol_filter(ydata, 21, 4))
    ln2.set_data(freq, maxhold)
    return ln,

ani = FuncAnimation(fig, update, frames=num_frames,
                    init_func=init, interval = period*sampwindow*1e3, blit=False, repeat = True)
plt.show()
# %%
