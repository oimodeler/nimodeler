# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:37:50 2025

@author: ame
"""

import nifits.backend as be
import nifits.io.oifits as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
from astropy.constants import h
from astropy.modeling import models
from astropy.io import fits
import oimodeler as oim
import time
from tqdm import tqdm
from matplotlib.colors import LogNorm
from pathlib import Path


#channel=["PHOT1","PHOT2","ADDI1","NULL1","NULL2","ADDI2","PHOT3","PHOT4"]

flux_unit = u.W / u.Hz / u.m**2 / u.sr

#def chan(name):
#    return channel.index(name)

class nimData(object):
    def __init__(self,fname,model=None,dim=None,fov=None):
        with fits.open(fname) as anhdu:
            self.nifit = io.nifits.from_nifits(anhdu)

        self.backend = be.NI_Backend(self.nifit)
        self.abe = be.NI_Backend()
        self.abe.add_instrument_definition(self.nifit)
        self.abe.create_fov_function_all()
        self.wl = self.abe.nifits.oi_wavelength.data_table["EFF_WAVE"].data

        if dim!=None and fov!=None:
            self.getResponse(dim,fov)

    def simultateResponse(self,dim,fov):
        self.dim= dim
        self.fov = fov
        self.hfov = fov/2
 
        halfrange_rad =  self.hfov *u.mas.to(u.rad)
        xs = np.linspace(-halfrange_rad, halfrange_rad, dim)
        self.xx, self.yy = np.meshgrid(xs, xs)
        
        self.map_extent = [- self.hfov ,  self.hfov , - self.hfov ,  self.hfov ]
        
        self.response = self.abe.get_all_outs(self.xx.flatten(), self.yy.flatten(), kernels=False)
        
        return self.response

    def plotFov(self):
        map_fov = self.abe.nifits.ni_fov.xy2phasor(self.xx.flatten(), self.yy.flatten())
        fig,ax = plt.subplots()
        ma= ax.imshow(np.abs(map_fov[0,0,:].reshape((self.xx.shape))), extent=self.map_extent)
        plt.colorbar(ma)
        ax.contour(np.abs(map_fov[0,0,:].reshape((self.xx.shape))), levels=(0.5,), extent=self.map_extent)
        ax.set_title("FOV ($\\lambda_0$)")
        return fig, ax
    
    def plotResponse(self):
        channel=self.getChannels()
        fig, ax = plt.subplots(2,4,figsize=(15,8))
        fig.suptitle("Response of outputs")
        ax = ax.flatten()
        for i in range(8):
            ax[i].imshow(np.abs(self.response[9,0,i,:].reshape((self.xx.shape))), extent=self.map_extent)
            ax[i].contour(np.abs(self.response[9,0,i,:].reshape((self.xx.shape))), levels=(0.5,), extent=self.map_extent)
            ax[i].text(0, self.hfov *0.95, f"{channel[i]}",va="top",ha="center",color="w")
            ax[i].plot(np.array([self.xx.min(),self.xx.max()])*u.rad.to(u.mas),[0,0],color="r",ls="--",lw=2,alpha=0.5)
            ax[i].plot([0,0],np.array([self.yy.min(),self.yy.max()])*u.rad.to(u.mas),color="r",ls="--",lw=2,alpha=0.5)              
            if i//4 == 0:
                ax[i].get_xaxis().set_visible(False)
            else:
                ax[i].set_xlabel("x [mas]")
                
            if i%4 == 0:
                ax[i].set_ylabel("y [mas]")
            else:
                ax[i].get_yaxis().set_visible(False)

            plt.tight_layout()
        return fig,ax
    
    
    def getChannels(self):
        
        arr     = self.nifit.ni_iotags
        is_add  = arr.outbright[0]
        is_phot = arr.outphot[0]
        is_null = arr.outdark[0]
        
        res=[]
        for i in range(is_add.size):
            resi=""
            if is_add[i]:
                resi+="ADDI_"
            if is_phot[i]:
                resi+="PHOT_"
            if is_null[i]:
                resi+="NULL_"
            res.append(resi[:-1])
        return(np.array(res))
        
    
    
            
                
        
        
    
    
  


#%%



def loadnifitsData(input):
   
    if isinstance(input, nimData):
        data = input
    else:
        if isinstance(input, (fits.hdu.hdulist.HDUList, str, Path)):
            input = [input]

        if isinstance(input, list):
            data = []

            for elem in input:
                if isinstance(elem, fits.hdu.hdulist.HDUList):
                    data.append(elem)
                else:
                    try:
                        data.append(fits.open(elem))
                    except:
                        raise ValueError(
                            "The path does not exist or is not a"
                            " valid fits files"
                        )
        else:
            raise TypeError(
                "Only nimData, hdulist, Path or string, or list of"
                " these kind of objects allowed "
            )

    return data

class nimSimulator(object):
    
    def __init__(self,data,model):
        
        self.simulatedData = np.array([])
        self.diffSimulatedData = np.array([])
        
        
        if data != None:
            if isinstance(data, nimData):
                self.data = data
            else:
                self.data = loadnifitsData(data)

        if model != None:
            self.setModel(model)
            
    def setModel(self, model):
        self.model = model

    def plotModel(self,dim=None,fov=None):
        if dim==None:
            dim=self.dim
        if fov==None:
            fov=self.fov
        pix=fov/dim
        map_extent = [-fov/2, fov/2, -fov/2, fov/2]
        im = np.flip(self.model.getImage(dim,pix,wl=self.wl,fromFT=False),axis=1)
        fig, ax = plt.subplots()
        cmap = ax.imshow(im[0,:,:],norm=mpl.colors.PowerNorm(gamma=0.1),extent=map_extent)
        fig.colorbar(cmap, ax=ax,label="Normalized flux")
        ax.set_title("oimodeler model image")
        return fig, ax
    
    def simulateData(self,dim=None,fov=None):
    
        if dim==None:
            dim = self.data.dim
        if fov==None:
            fov = self.data.fov
            
        #nframe = self.response.shape[0]
    
        pix=fov/dim
        
        im = np.flip(self.model.getImage(dim,pix,wl=self.data.wl,fromFT=False),axis=1)
       
        resp=self.data.response.reshape(self.data.response.shape[0],
                                        self.data.response.shape[1],
                                        self.data.response.shape[2],dim,dim)
        
        im1 = im[np.newaxis,:,np.newaxis,:,:]
        outputs = np.abs(resp*im1)
        
        simul = np.sum(outputs,axis=(-1,-2))
        self.simulatedData = simul+6312 # offset for cold bkg JS

    
        #return self.simulatedData 
       
    
    def getDiffSimulatedData(self,dim=None,fov=None,recompute=False):
        if self.diffSimulatedData.size==0 or recompute==True:
            _ = self.simulateData(dim,fov)
            
        chan = self.data.getChannels()
        nulls =np.where(chan == "NULL")[0]
        self.diffSimulatedData = self.simulatedData[:,:,nulls[0]]-self.simulatedData[:,:,nulls[1]]
        return self.diffSimulatedData 
    
    def getDiffSimulatedDataOnly(self,dim=None,fov=None):
        
        if dim==None:
            dim = self.dim
        if fov==None:
            fov = self.fov
            
        pix=fov/dim
        
        im = np.flip(self.model.getImage(dim,pix,wl=self
                                         .wl,fromFT=False),axis=1)
        chan = self.data.getChannels()
        nulls =np.where(chan == "NULL")[0]
        resp1=self.response[:,:,nulls[0],:].reshape(self.response.shape[0],self.response.shape[1],dim,dim)
        resp2=self.response[:,:,nulls[1],:].reshape(self.response.shape[0],self.response.shape[1],dim,dim)        
    
        im1 = im[np.newaxis,:,:,:]
        outputs1 = np.abs(resp1*im1)
        outputs2 = np.abs(resp2*im1)
        
        simul1 = np.sum(outputs1,axis=(-1,-2))
        simul2 = np.sum(outputs2,axis=(-1,-2))       
    
        self.diffSimulatedData = simul1-simul2
        return self.diffSimulatedData 
        
    def getDiffSimulatedDataOnly2(self,dim=None,fov=None):
        
        if dim==None:
            dim = self.data.dim
        if fov==None:
            fov = self.data.fov
            
        pix=fov/dim    
    
        im = np.flip(self.model.getImage(dim,pix,wl=self.data.wl,fromFT=False),axis=1)
        
        chan = self.data.getChannels()
        nulls =np.where(chan == "NULL")[0]
        
        resp1=self.data.response[:,:,nulls[0],:]
        resp2=self.data.response[:,:,nulls[1],:]
        
        nwl = self.data.wl.size
        nframe = self.data.response.shape[0]
        
        simul1 = np.ndarray((nframe,nwl))
        simul2 = np.ndarray((nframe,nwl))       
        
        for iwl in range(nwl):
            
            imi = im[iwl,:,:].flatten()
            
            idx = np.where(imi!=0)
            
            imidx = np.take(imi,idx)
            
            outputs1=np.ndarray((nframe,idx[0].size))
            outputs2=np.ndarray((nframe,idx[0].size))   
            
            for iframe in range(nframe):
            
                outputs1[iframe,:] = np.abs(np.take(resp1[iframe,iwl,:],idx)*imidx)
                outputs2[iframe,:] = np.abs(np.take(resp2[iframe,iwl,:],idx)*imidx)
        
            simul1[:,iwl] = np.sum(outputs1,axis=(-1))
            simul2[:,iwl] = np.sum(outputs2,axis=(-1))       
    
        self.diffSimulatedData = simul1-simul2
        return self.diffSimulatedData 
            
    def getData(self):
        return self.data.abe.nifits.ni_iout.iout
    
    def getDiffData(self):
        return self.data.abe.nifits.ni_kiout.kiout[:,:,0]
    
    def getError(self):
        data_cov = self.data.abe.nifits.ni_kcov.kcov
        return np.sqrt(np.diagonal(data_cov,axis1=1,axis2=2))
    
    def computeChi2(self):
        simData =self.getDiffSimulatedDataOnly2()#recompute=True)
        data = self.getDiffData()
        err = self.getError()
        
        chi2 = np.sum((simData-data)**2 / err**2)
        nel = np.size(err)
        
        nparam = 1 #len(self.model.getFreeParameters())
        chi2r = chi2/(nel - nparam)
        return chi2,chi2r,nel
    
    def plotDiffrential(self,iFrame=0,ax=None,showLabel=True,recompute=True):
        if  self.diffSimulatedData.size==0 or recompute==True:
            _ = self.getDiffSimulatedData(recompute=True)
       
        data_cov = self.data.abe.nifits.ni_kcov.kcov
        err = np.sqrt(np.diagonal(data_cov,axis1=1,axis2=2))
        
        if ax==None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            
        
        data_diff_outputs = self.data.abe.nifits.ni_kiout.kiout
        ax.errorbar(self.data.wl*1e6,data_diff_outputs[iFrame,:,0],err[iFrame,:],label="data from the niftis")
        
        ax.plot(self.data.wl*1e6,self.diffSimulatedData[iFrame,:],label="model from oimodeler")
        if showLabel:
            ax.legend()
        #plt.title("Differential null flux")
        
        return fig, ax
    
    def plot(self,iFrame):
        
        simData =  self.simulatedData
        data = self.getData()
        iphot=0
        iaddi=0
        inull=0
        def col(name,iphot,iaddi,inull):
            #global iphot,iaddi,inull
            if name == "PHOT":
                cols = np.array(["k","dimgray","gray","silver"])
                i=iphot
                iphot+=1
            elif name == "ADDI":
                cols = np.array(["orange","gold","goldenrod","darkorange"])
                i=iaddi
                iaddi+=1
            elif name == "NULL":
                cols = np.array(["tab:green","tab:blue"])
                i=inull
                inull+=1
            return cols[i],iphot,iaddi,inull
                
        #cols=["k","k","grey","g","r","grey","k","k"]
        
        fig, ax = plt.subplots()
        channel = self.data.getChannels()
        for iout in range(8):
            c,iphot,iaddi,inull=col(channel[iout],iphot,iaddi,inull)
            ax.plot(self.data.wl*1e6,data[iFrame,:,iout],label=(f"{channel[iout]}"),color=c)
            ax.plot(self.data.wl*1e6,simData[iFrame,:,iout],ls="--",color=c)
        
        ax2 = ax.twinx()
        ax2.plot(np.NaN, np.NaN, ls="-",label='nifits data', c='k')
        ax2.plot(np.NaN, np.NaN, ls="--",label='oimodeler', c='k')
        ax2.get_yaxis().set_visible(False)
        
        ax.legend(loc=1)
        ax2.legend(loc=3)
        plt.title("Flux for the channels from the nifits files")
        ax.set_xlabel("$\lambda$ ($\mu$m)")
        ax.set_ylabel("Flux (unit?)")
        ax.set_yscale("log")
        
        return fig, ax

