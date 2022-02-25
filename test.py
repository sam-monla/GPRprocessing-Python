from hashlib import new
import os
from turtle import position
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import file_mgmt as fmgmt
import basic_proc as bp
import phasePicking_tools as phaseTools
import deconvolution as decon
from tqdm import tqdm

head, dat = fmgmt.readDZT('/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR/GRID____088.3DS/FILE____001.DZT', [2])

start,end = bp.rem_empty(dat)
dat = dat[:,start:end]
dat = bp.deWOW(dat,18)
dat2 = bp.rem_mean_trace(dat,dat.shape[0])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
maxi2 = 32768/2
mini2 = -32768/2
plt.imshow(dat2, cmap='bwr', vmin=mini2, vmax=maxi2)
ax.set_aspect(8)
plt.title("Radargramme original")
plt.xlabel("Traces")
plt.ylabel("Échantillons")
#plt.show()

newdata = np.zeros(dat2.shape)
for tr in tqdm(range(dat2.shape[1])):
    dr, DR = decon.DR_spectral_enhancement(dat2[:,tr],0.273)
    DR = np.reshape(DR,510)
    newdata[:,tr] = DR

print(np.amax(newdata))
print(np.amin(newdata))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
maxi3 = 12
mini3 = -12
plt.imshow(newdata, cmap='bwr', vmin=mini3, vmax=maxi3)
ax.set_aspect(8)
plt.title("Amélioration de la résolution spectrale")
plt.xlabel("Traces")
plt.ylabel("Échantillons")
plt.show()

