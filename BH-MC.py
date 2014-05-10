__author__ = 'Abuenameh'

import os
import sys
import pyalps
import gtk
import gobject
import threading
import datetime
import itertools
import concurrent.futures
import numpy as np
from mathematica import mathformat
from switch import switch
from speed import gprogress

numthreads = 15

L = 2
nmax = 7
T = 0.005
thermalization = 10000
sweeps = 500000

if len(sys.argv) < 3:
    print('Insufficient number of command line arguments.')
    quit(1)

seed = int(sys.argv[3])

delta = float(sys.argv[2])
if delta == 0:
    lattice = "square lattice"
else:
    lattice = "inhomogeneous square lattice"

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
filenameprefix = 'BH_MC_'

measurements = ['Energy', 'Stiffness', 'Density', 'Density^2', 'Local Density', 'Local Density^2']

nu = 0
if delta > 0:
    if sys.platform == 'darwin':
        getnu = 'delta*2*(random()-0.5)'
    elif sys.platform == 'linux2':
        np.random.seed(int(sys.argv[3]))
        nu = delta*2*np.random.random(L*L) - delta
        getnu = 'get(' + str(L) + '*x+y,' + ",".join([str(nui) for nui in nu]) + ')'
else:
    getnu = ''


parmsbase = {
            'TEMP_DIRECTORY': bhdir,
            'MEASURE[Density]': 1,
            'MEASURE[Density^2]': 1,
            'MEASURE[Local Density]': 1,
            'MEASURE_LOCAL[Local Density^2]': "n2",
            'LATTICE'        : lattice,
            'MODEL'          : "boson Hubbard",
            'T'              : T,
            'L'              : L,
            'U'              : 1.0,
            'delta'          : delta,
            'NONLOCAL'       : 0,
            'Nmax'           : nmax,
            'DISORDERSEED'   : 12345,
            'THERMALIZATION' : thermalization,
            'SWEEPS'         : sweeps
        }

parms = [parmsbase]

def runmc(i, t, mu, it, imu):
    # parms = [dict(parmsbase.items() + { 't': t, 'mu': str(mu) + '-' + getnu, 'it': it, 'imu': imu}.items())]
    parms = [dict(parmsbase.items() + { 't': t, 'mu': str(mu), 'it': it, 'imu': imu}.items())]
    input_file = pyalps.writeInputFiles(filenameprefix + str(i), parms)
    pyalps.runApplication('/opt/alps/bin/worm', input_file, writexml=True, Tmin=5)


def runmain():
    beta = 1.0/T

    ts = np.linspace(0.01, 0.3, 1).tolist()
    mus = np.linspace(0, 1, 0.5).tolist()
    ts = [ 0.1]
    mus = [0.5]
    # ts = [np.linspace(0.01, 0.3, 10).tolist()[2]]
    # ts = [0.3]
    # ts = np.linspace(0.3, 0.3, 1).tolist()

    dims = [len(ts), len(mus)]
    ndims = dims + [L]

    E0res = np.zeros(dims)
    fsres = np.zeros(dims)
    nres = np.zeros(dims)
    n2res = np.zeros(dims)
    kres = np.zeros(dims)
    nires = np.zeros(ndims)
    n2ires = np.zeros(ndims)
    kires = np.zeros(ndims)

    E0res.fill(np.nan)
    fsres.fill(np.nan)
    nres.fill(np.nan)
    n2res.fill(np.nan)
    kres.fill(np.nan)
    nires.fill(np.nan)
    n2ires.fill(np.nan)
    kires.fill(np.nan)

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmc, i, tmu[0][0], tmu[0][1], tmu[1][0], tmu[1][1]) for i, tmu in
                   enumerate(zip(itertools.product(ts, mus), itertools.product(range(0, len(ts)), range(0, len(mus)))))]
        for future in gprogress(concurrent.futures.as_completed(futures), size=len(futures)):
            pass

    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix), measurements)
    for d in data:
        for s in d:
            it = int(s.props['it'])
            imu = int(s.props['imu'])
            for case in switch(s.props['observable']):
                if case('Energy'):
                    E0res[it][imu] = s.y[0]
                    break
                if case('Stiffness'):
                    fsres[it][imu] = s.y[0]
                    break
                if case('Density'):
                    nres[it][imu] = s.y[0]
                    break
                if case('Density^2'):
                    n2res[it][imu] = s.y[0]
                    break
                if case('Local Density'):
                    nires[it][imu] = s.y[0]
                    break
                if case('Local Density^2'):
                    n2ires[it][imu] = s.y[0]
                    break
    for it, imu in itertools.product(ts, mus):
        kres[it][imu] = beta*(n2res[it][imu] - nres[it][imu]**2)
        kires[it][imu] = beta*(n2ires[it][imu] - nires[it][imu]**2)

    end = datetime.datetime.now()

    resi = sys.argv[1]
    if sys.platform == 'darwin':
        resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-MC/res.' + str(resi) + '.txt'
    elif sys.platform == 'linux2':
        resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-MC/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    res = ''
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'Tres[{0}]={1};\n'.format(resi, T)
    res += 'thermres[{0}]={1};\n'.format(resi, thermalization)
    res += 'sweepsres[{0}]={1};\n'.format(resi, sweeps)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'nures[{0}]={1};\n'.format(resi, mathformat(nu))
    res += 'mures[{0}]={1};\n'.format(resi, mathformat(mus))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'fsres[{0}]={1};\n'.format(resi, mathformat(fsres))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'kres[{0}]={1};\n'.format(resi, mathformat(kres))
    res += 'nires[{0}]={1};\n'.format(resi, mathformat(nires))
    res += 'n2ires[{0}]={1};\n'.format(resi, mathformat(n2ires))
    res += 'kires[{0}]={1};\n'.format(resi, mathformat(kires))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)
    resf.write(res)

    print '{0}\n'.format(mathformat(nres))
    print '{0}\n'.format(mathformat(n2res))

    gtk.main_quit()

def startmain():
    main_thread = threading.Thread(target=runmain);
    main_thread.start()
    return False

if __name__ == '__main__':
    os.chdir(bhdir)
    [os.remove(f) for f in os.listdir(".")]
    gobject.timeout_add(1000, startmain)
    gtk.gdk.threads_init()
    gtk.main()



os.chdir(bhdir)
input_file = pyalps.writeInputFiles(filenameprefix,parms)

res = pyalps.runApplication('/opt/alps/bin/worm',input_file,Tmin=5)
# data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix=filenameprefix), measurements)
data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix=filenameprefix),['Energy'])
# print data

# for d in data:
#     for s in d:
        # print s.props['observable']

# res = pyalps.runApplication('/opt/alps/bin/sparsediag',input_file,Tmin=5)
# data2 = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix),['Energy'])

# print data[0][0].y
# print data2[0][0][0].y

