#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as sg
import sys
import os
# %%
if (len(sys.argv) != 2):
    print("Invocation: %s <Binary File>"%(sys.argv[0]))
    sys.exit(0)
fname = sys.argv[1]
basename = os.path.basename(fname)
dirname = os.path.dirname(fname)
outputname = dirname + basename + '.mp4'
words = basename.split('_')
date = words[0][0:4] + "-" + words[0][4:6] + "-" + words[0][6:8]
start_time = words[1][0:2] + ":" + words[1][2:4] + ":" + words[1][4:6]
passlength = words[2] + "s"
norad_id = words[3]
# %%
data = np.fromfile(fname, dtype = np.single, count = -1)
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
t_win_ = 20
t_win = int(t_win_/(sampwindow*period)) # 20 seconds == these many lines
t_scale = 1 # int(sampwindow/t_win) # how many extra lines per line
t_scale = 1 if t_scale < 1 else t_scale
print("Time scale = %d, window height = %d"%(t_scale, t_win))
image = np.zeros((sampwindow, t_win * t_scale), dtype = float) - 140 # 140 dB
extent = [freq.min(), freq.max(), -t_win_, 0]
# %%
# Set up max hold
maxhold = np.zeros(freq.shape, dtype = float) - np.inf
minhold = np.zeros(freq.shape, dtype = float) + np.inf
# %%
fig, ax = plt.subplots()
im = ax.imshow(image.transpose(), origin = 'upper', extent=extent, animated = True, vmin = -50, vmax = -10, aspect = 'auto')
fig.colorbar(im)
ax.set_xlim(cfreq - 0.5 * bw, cfreq + 0.5 * bw)

def init():
    im.set_data(image.transpose())
    return [im]

def update(frame):
    fig.suptitle("Date: %s, Start: %s, NORAD ID: %s, Length: %s\nTime: %.2f/%.2f s"%(date, start_time, norad_id, passlength, (frame + 1)*period*sampwindow, num_frames*period*sampwindow))
    ydata = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(bbdata[frame*sampwindow:(frame+1)*sampwindow], norm="ortho"))))
    # 1. Move old frames "up"
    for i in range(t_win - 2, -1, -1):
        # print("Moving %d:%d to %d:%d"%(i*t_scale, (i+1)*t_scale - 1, (i+1)*t_scale, (i+2)*t_scale - 1))
        image[:, (i+1)*t_scale:(i+2)*t_scale] = image[:, i*t_scale:(i+1)*t_scale]
    # 2. Copy new frame
    for i in range(t_scale):
        image[:, i] = ydata
    # 3. Show new frame
    im.set_array(image.transpose())
    return [im]

ani = FuncAnimation(fig, update, frames=num_frames, interval = period*sampwindow*1e3/10, blit=False, repeat = True)
ani.save(outputname)
# %%
