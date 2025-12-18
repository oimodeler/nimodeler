# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:39:21 2025

@author: ame
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.units as u
from astropy.constants import h
from astropy.modeling import models
import oimodeler as oim
import nimodeler as nim
import numpy as np
from pathlib import Path
from tqdm import tqdm


dir0=Path(__file__).parents[1]
dirdata=dir0 / "data"
dirplot=dir0 / "images"

#fname = dir0 / 'nott_ut_planet_10_2.nifits'
fname = dirdata / 'nobackground_v3.nifits'

data=nim.nimData(fname)

#%%
dim = 101 # pixels
fov = 40 # in mas
resp = data.simultateResponse(dim, fov)

#%% plot the field of view and trasnmission responses of the nuller
fig_resp, ax_resp = data.plotResponse()
fig_resp.savefig(dirplot / f"test_{fname.stem}_channels_responses.png")

fig_fov, ax_fov = data.plotFov()
fig_fov.savefig(dirplot / f"test_{fname.stem}_fov.png")


#%% create star planet model with two blackbodies
try:
    d = data.nifit.header['SCIFYSIM DISTANCE'] * u.pc
    Tstar   = data.nifit.header['SCIFYSIM T_STAR'] * u.K 
    Rstar   = data.nifit.header['SCIFYSIM R_STAR'] * u.Rsun
    Tplanet = data.nifit.header['SCIFYSIM T_PLANET'] * u.K
    Rplanet = data.nifit.header['SCIFYSIM R_PLANET']* u.Rsun
    sep = data.nifit.header['SCIFYSIM SEP'] 
    pa = data.nifit.header['SCIFYSIM PA'] 
    x = sep*np.sin(np.deg2rad(pa))
    y = sep*np.cos(np.deg2rad(pa))
    
except:
    Tplanet = 1000 * u.K
    Rplanet = 30 * u.Rjup # 60, 30, 3
    Tstar   = 5300 * u.K 
    Rstar   = 0.7 * u.Rsun
    d = 10.8 * u.pc
    x=-17.85
    y=-0.0


Tback   = 300.0 * u.K

wl = data.nifit.oi_wavelength.lambs
flux_unit = u.W / u.Hz / u.m**2 / u.sr

planet_sr = np.pi * (Rplanet.to(u.m) / d.to(u.m))**2 
star_sr = np.pi * (Rstar.to(u.m) / d.to(u.m))**2 

fovAt=250*3.5/1.6
bckg_sr = ((fov*u.mas/2)**2*np.pi).to(u.sr)

bb_planet = models.BlackBody(temperature=Tplanet)
bb_star   = models.BlackBody(temperature=Tstar)
bb_bckg   =  models.BlackBody(temperature=Tback)

energy_per_photon = ((wl*u.m).to(u.Hz, u.spectral()) * h).value * u.J / u.photon
d_freq = np.abs(np.gradient((wl*u.m).to(u.Hz, u.spectral())))

f0=1.0

fplanet = bb_planet(wl*u.m) * f0
fplanet = fplanet.to(flux_unit) / energy_per_photon * planet_sr * d_freq
fstar   = bb_star(wl*u.m) * f0 
fstar = fstar.to(flux_unit) / energy_per_photon * star_sr * d_freq
fback   = bb_bckg(wl*u.m) * f0 *0
fback = fback.to(flux_unit) / energy_per_photon*bckg_sr * d_freq

star   = oim.oimPt(x=0,y=0,f=oim.oimInterp("wl",wl=wl,values=fstar.value))
planet = oim.oimPt(x=x,y=y,f=oim.oimInterp("wl",wl=wl,values=fplanet.value))
bckg   = oim.oimBackground(f=oim.oimInterp("wl",wl=wl,values=fback.value))
model  = oim.oimModel(star,planet,bckg)

#%%

fig,ax=plt.subplots()
ax.plot(wl*1e6,fplanet,label="planet",marker=".")
ax.plot(wl*1e6,fstar,label="star",marker=".")
ax.plot(wl*1e6,fback,label="background",marker=".")
ax.legend()
ax.set_xlabel("wavelength ($\mu$m)")
ax.set_ylabel("Flux (arbitrary unit)")
ax.set_yscale("log")
fig.suptitle(fname.name)
fig.savefig(dirplot / f"flux_component_{fname.stem}.png")
#%% plot the model
pix = fov/dim
fig,ax,im  = model.showModel(dim,pix,wl=3.5e-6,normPow=0.1)
fig.savefig(dirplot / f"image_model0_{fname.stem}.png")
#%% create the simulator cointaining the data and the model
sim  =nim.nimSimulator(data,model)
sim.simulateData()
#%% plot a data/model comparison for one frame

fig, ax = sim.plot(10)

isbackground = fback.min().value !=0

if isbackground:
    txtBckG=f"{Tback}"
else:
    txtBckG="None"

fig.suptitle(f"{fname.name} - Bckg = {txtBckG}")
fig.savefig(dirplot / f"flux_allchannels_{fname.stem}.png")
#%% plot the differential data 
fig,ax = plt.subplots(4,5,figsize=(16,10))
ax = ax.flatten()
for iFrame in range(sim.getData().shape[0]):
    _,_ = sim.plotDiffrential(iFrame,ax=ax[iFrame],showLabel=iFrame==0)
fig.suptitle(fname.name)
fig.savefig(dirplot / f"diff_channel_flux_{fname.stem}.png")

#%%
dim=41
chi2map0 = np.ndarray((dim,dim))
xy=np.linspace(-20,20,dim)

for i,x in enumerate(tqdm(xy)):
    for j,y in enumerate(xy):
        planet.params['x'].value=x
        planet.params['y'].value=y
        sim.simulateData()

        chi2map0[j,i]=sim.computeChi2()[1]
#%%
dxy=np.gradient(xy)[0]/2*0

chi2map0/=chi2map0.min()
fig,ax=plt.subplots()
cb=ax.imshow(chi2map0,extent=[xy[0]-dxy,xy[-1]-dxy,xy[0]-dxy,xy[-1]]-dxy,
             interpolation="None",norm=colors.LogNorm())
plt.colorbar(cb,label="$\\chi^2_r$")
ax.set_xlabel("$\\alpha$ [mas]")
ax.set_ylabel("$\\delta$ [mas]")

xy0=np.where(chi2map0==chi2map0.min())
x0=xy[xy0[1][0]]
y0=xy[xy0[0][0]]
ax.plot([xy[0],xy[-1]],[y0,y0],ls=":",color="r")
ax.plot([x0,x0],[xy[0],xy[-1]],ls=":",color="r")
fig.savefig(dirplot / f"position_exploration_{fname.stem}.png")


        