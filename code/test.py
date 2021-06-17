import numpy as np
import matplotlib.pyplot as mpl
from scipy.signal import (sosfilt, sosfiltfilt, butter)
import clean

f = [.02, 1, 50]
sr = 0.00001
t = np.arange(0, .25, sr)
nt = len(t)
s = np.zeros((len(f),nt))
for j in range(nt):
   for i in range(len(f)):
      s[i,j] = np.sin(2*np.pi*t[j]/f[i])
st = s[0,:]/5+s[1,:]+s[2,:]*10

sf1 = sosfiltfilt(butter(2, [100*sr,150*sr], analog=False, btype='band', output='sos'), st)
sf2 = sosfiltfilt(butter(5, [100*sr,150*sr], analog=False, btype='band', output='sos'), st)
sf3 = sosfiltfilt(butter(8, [100*sr,150*sr], analog=False, btype='band', output='sos'), st)
sf4 = sosfiltfilt(butter(10, [100*sr,150*sr], analog=False, btype='band', output='sos'), st)
#sf2 = sosfiltfilt(butter(5, [0.5*sr,5*sr], analog=False, btype='band', output='sos'), st)
#sf3 = sosfiltfilt(butter(5, [0.001*sr,.1*sr], analog=False, btype='band', output='sos'), st)

mpl.figure(figsize = (16,8))
#mpl.plot(t,st,'k')
mpl.plot(t,s[0,:]/5,'b')
#mpl.plot(t,s[1,:],'g')
#mpl.plot(t,s[2,:]*10,'b')
mpl.plot(t,sf1,'--r')
mpl.plot(t,sf2,'r')
mpl.plot(t,sf3,'--g')
mpl.plot(t,sf4,'g')
#mpl.plot(t,sf2,'--g')
#mpl.plot(t,sf3,'--b')
mpl.show()


