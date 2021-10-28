#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm as cmap
import numpy as np
import sys

# Roy: just qa==0 are not good

print(' args', len(sys.argv) , sys.argv)
if len(sys.argv)!=2 : 
    print('provide input YAML name, aborting'); exit(1)
h5N=sys.argv[1]

# parNameLF=[ ('gnabar',0), ('gkbar',1), ('gcabar',2),('gl',3),  ('cm',4) ] # HH_5par_v1
# parNameLF = [ ('a', 0), ('b', 1), ('c', 2), ('d', 3) ]
# parNameLF = [ ('gnabar_soma', 0), ('gnabar_dend', 1), ('gkbar_soma', 2), ('gkbar_dend', 3), ('gcabar_soma', 4), ('gcabar_dend', 5), ('gl_soma', 6), ('gl_dend', 7), ('cm', 8) ] ## HH 9param
parNameLF = [ ('gnabar_soma', 0), ('gnabar_apic', 1), ('gnabar_basal', 2), ('gkbar_soma', 3), ('gkbar_apic', 4), ('gkbar_basal', 5), ('gcabar_soma', 6), ('gcabar_apic', 7), ('gcabar_basal', 8), ('gl_soma', 9), ('gl_apic', 10), ('gl_basal', 11), ('cm', 12) ] ## HH 13params
#Respectively, those are max sodium conductance, max potassium conductance, max calcium conductance, leak conductance, and membrane capacitance.

print('Opening ',h5N)
# change filepath if neessary
h5f = h5py.File(h5N, 'r')
print('see keys number:',len(h5f))
for x in h5f.keys():
  print('key=',x, end=' ')
  y=h5f[x]
  print(', size:',y.shape)

voltB=h5f['voltages']
physRange=np.array(h5f['phys_par_range'])

nTrace,nBins=voltB.shape
print('num frames=', nTrace,type(voltB))
#print('e.g. uPar:',h5f['norm_par'][0,:])

# repack params to keep only varied ones

listU=[]; listP=[]
parNameL=[]
for (name,idx) in parNameLF:
    #print('get',name,idx,' physRange:',physRange[idx,:])    
    print('get',name,idx)
    listU.append(np.array(h5f['norm_par'][:,idx]))
    listP.append(np.array(h5f['phys_par'][:,idx]))
    parNameL.append(name)
assert len(listU) >0
parU=np.stack(tuple(listU),axis=-1 )
parP=np.stack(tuple(listP),axis=-1 )
print('parU shape',parU.shape)
print('physRange shape',physRange.shape,physRange)


stim=h5f['stim']
stim=np.array(stim)

nPar=parP.shape[1]
print('nBins=',nBins,' nPar=',nPar, parNameL)

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# plotter # select few random traces
plt.figure(facecolor='white', figsize=(13,6))
nrow,ncol=3,3
j=0

for ii in range(50):  
    i=np.random.randint(nTrace)
    i=ii
    voltage = voltB[i]#-voltB[0]
    id = i
    paru=parU[i]
    parp=parP[i]
    print('i','upar',paru)

    ax = plt.subplot(nrow, ncol, 1+j)
    #print('i=',i,pars,id,parp)
    j+=1
    
    ax.plot(voltage)
    ax.plot(stim*2.)

    parTxt=[ '%.2f, '%x for x in paru]
    ax.set(title="U:%s"%(''.join(parTxt)),xlabel='time bin ', ylabel='trace:%d'%id)
    #ax.set_xlim(10000,17600)
    if j>=ncol*nrow : break

plt.tight_layout()

#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
# plotter, correaltion between params

plt.figure(facecolor='white', figsize=(10,6))
nrow,ncol=3,nPar

mm=1.1
binsX= np.linspace(-mm,mm,30)

for j in range(nPar):
    ax = plt.subplot(nrow, ncol, 1+j)
    ax.hist(parU[:,j],bins=50)

    ax = plt.subplot(nrow, ncol, 1+j+ncol)
    ax.scatter(parU[:,j],parP[:,j])
    ax.set(title=parNameL[j],xlabel='Upar-%d'%j,ylabel='phys value')

    ax = plt.subplot(nrow, ncol, 1+j+2*ncol)
    j1=(j+1)%nPar
    #ax.scatter(parU[:,j],parU[:,j1])
    y1=parU[:,j]
    y2=parU[:,j1]
    zsum,xbins,ybins,img = ax.hist2d(y1,y2,bins=binsX, cmin=1,
                                   cmap = cmap.rainbow)    

    plt.colorbar(img, ax=ax)

    ax.set(title='corr Upar',xlabel='Upar-%d'%j,ylabel='Upar-%d'%j1)


    print('Ppar%d  %s physRange: %.3g , %.3g '%(j,parNameL[j],min(parP[:,j]),max(parP[:,j])),end='')
    print('Urange: %.3g , %.3g'%(min(parU[:,j]),max(parU[:,j])))


plt.tight_layout()
plt.show()
