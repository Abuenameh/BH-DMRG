__author__ = 'Abuenameh'

import os
import sys
import threading
import datetime
import itertools
import xml.etree.ElementTree as ET

import pyalps
import gtk
import gobject
import concurrent.futures
import numpy as np
from mathematica import mathformat
from switch import switch
from speed import gprogress


numthreads = 5

L = 60
nmax = 7
T = 0.01
thermalization = 10000
# thermalization = 1000000
sweeps = 500000
# sweeps = 10000000
# limit = 600
# limit=0
limit=600
app="worm"
# app="dirloop_sse"

beta = 1.0 / T

d = 2

if d == 1:
    numsites = L
elif d == 2:
    numsites = L * L

if len(sys.argv) < 3:
    print('Insufficient number of command line arguments.')
    quit(1)

seed = int(sys.argv[3])

delta = float(sys.argv[2])
if delta == 0:
    if d == 1:
        lattice = "open chain lattice"
    elif d == 2:
        lattice = "square lattice"
else:
    lattice = "inhomogeneous square lattice"

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
filenameprefix = 'BH_MC_'

measurements = ['Energy', 'Stiffness', 'Density', 'Density^2', 'Local Density', 'Local Density * Global Density']

nu = 0
if delta > 0:
    if sys.platform == 'darwin':
        getnu = 'delta*2*(random()-0.5)'
    elif sys.platform == 'linux2':
        np.random.seed(int(sys.argv[3]))
        nu = delta * 2 * np.random.random(L * L) - delta
        getnu = 'get(' + str(L) + '*x+y,' + ",".join([str(nui) for nui in nu]) + ')'
else:
    getnu = '0'

parmsbase = {
    'TEMP_DIRECTORY': bhdir,
    'MEASURE[Density]': 1,
    'MEASURE[Density^2]': 1,
    'MEASURE[Local Density]': 1,
    'MEASURE[Local Compressibility]': 1,
    'LATTICE': lattice,
    'MODEL': "boson Hubbard",
    'T': T,
    'L': L,
    'U': 1.0,
    'delta': delta,
    'NONLOCAL': 0,
    'Nmax': nmax,
    'DISORDERSEED': 12345,
    'THERMALIZATION': thermalization,
    'SWEEPS': sweeps
}

parms = [parmsbase]


def runmc(i, t, mu, it, imu):
    parms = [dict(parmsbase.items() + {'t': t, 'mu': str(mu) + '-' + getnu, 'it': it, 'imu': imu}.items())]
    input_file = pyalps.writeInputFiles(filenameprefix + str(i), parms)
    if limit > 0:
        pyalps.runApplication('/opt/alps/bin/'+app, input_file, writexml=True, Tmin=5, T=limit)
    else:
        pyalps.runApplication('/opt/alps/bin/'+app, input_file, writexml=True, Tmin=5)


def runmain():
    ts = np.linspace(0.01, 0.08, 15).tolist()
    mus = np.linspace(0, 1, 51).tolist()
    # mus = np.linspace(0, 1, 51).tolist()
    # mus = np.linspace(0, 0.25, 15).tolist()
    ts = [0.01]
    # mus = [mus[1]]
    mus = mus[0:10]
    # ts = [ts[0]]
    # ts = np.linspace(0.01, 0.08, 15).tolist()
    # mus = [0.5]
    # ts = [np.linspace(0.01, 0.3, 10).tolist()[2]]
    # ts = [0.3]
    # ts = np.linspace(0.3, 0.3, 1).tolist()

    dims = [len(ts), len(mus)]
    ndims = dims + [numsites]

    finished = np.empty(dims, dtype=bool)

    E0res = np.empty(dims, dtype=object)
    fsres = np.empty(dims, dtype=object)
    nres = np.empty(dims, dtype=object)
    n2res = np.empty(dims, dtype=object)
    kres = np.empty(dims, dtype=object)
    nires = np.empty(ndims, dtype=object)
    ninres = np.empty(ndims, dtype=object)
    kires = np.empty(ndims, dtype=object)

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmc, i, tmu[0][0], tmu[0][1], tmu[1][0], tmu[1][1]) for i, tmu in
                   enumerate(zip(itertools.product(ts, mus), itertools.product(range(0, len(ts)), range(0, len(mus)))))]
        for future in gprogress(concurrent.futures.as_completed(futures), size=len(futures)):
            pass

    data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix=filenameprefix), measurements)
    for d in data:
        it = int(d[0].props['it'])
        imu = int(d[0].props['imu'])
        outfile = d[0].props['filename'][0:-12] + 'out.xml'
        tree = ET.parse(outfile)
        root = tree.getroot()
        finished[it][imu] = root[0].attrib['status'] == 'finished'
        for s in d:
            for case in switch(s.props['observable']):
                if case('Energy'):
                    E0res[it][imu] = s.y[0]
                    break
                if case('Stiffness'):
                    fsres[it][imu] = L * s.y[0]
                    break
                if case('Density'):
                    nres[it][imu] = s.y[0]
                    break
                if case('Density^2'):
                    n2res[it][imu] = s.y[0]
                    break
                if case('Local Density'):
                    nires[it][imu] = s.y
                    break
                if case('Local Density * Global Density'):
                    ninres[it][imu] = s.y
                    break
        kres[it][imu] = beta * (n2res[it][imu] - numsites * (nres[it][imu] ** 2))
        kires[it][imu] = beta * (ninres[it][imu] - nires[it][imu] * nres[it][imu])

    end = datetime.datetime.now()

    resi = sys.argv[1]
    if sys.platform == 'darwin':
        resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-MC/res.' + str(resi) + '.txt'
    elif sys.platform == 'linux2':
        resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-MC/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    res = ''
    res += 'finished[{0}]={1};\n'.format(resi, mathformat(finished))
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    # res += 'dres[{0}]={1};\n'.format(resi, d)
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'Tres[{0}]={1};\n'.format(resi, T)
    res += 'thermres[{0}]={1};\n'.format(resi, thermalization)
    res += 'sweepsres[{0}]={1};\n'.format(resi, sweeps)
    res += 'limitres[{0}]={1};\n'.format(resi, limit)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'nures[{0}]={1};\n'.format(resi, mathformat(nu))
    res += 'mures[{0}]={1};\n'.format(resi, mathformat(mus))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'E0res[{0}]={1:mean};\n'.format(resi, mathformat(E0res))
    res += 'E0reserr[{0}]={1:error};\n'.format(resi, mathformat(E0res))
    res += 'fsres[{0}]={1:mean};\n'.format(resi, mathformat(fsres))
    res += 'fsreserr[{0}]={1:error};\n'.format(resi, mathformat(fsres))
    res += 'nres[{0}]={1:mean};\n'.format(resi, mathformat(nres))
    res += 'nreserr[{0}]={1:error};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1:mean};\n'.format(resi, mathformat(n2res))
    res += 'n2reserr[{0}]={1:error};\n'.format(resi, mathformat(n2res))
    res += 'kres[{0}]={1:mean};\n'.format(resi, mathformat(kres))
    res += 'kreserr[{0}]={1:error};\n'.format(resi, mathformat(kres))
    res += 'nires[{0}]={1:mean};\n'.format(resi, mathformat(nires))
    res += 'nireserr[{0}]={1:error};\n'.format(resi, mathformat(nires))
    res += 'ninres[{0}]={1:mean};\n'.format(resi, mathformat(ninres))
    res += 'ninreserr[{0}]={1:error};\n'.format(resi, mathformat(ninres))
    res += 'kires[{0}]={1:mean};\n'.format(resi, mathformat(kires))
    res += 'kireserr[{0}]={1:error};\n'.format(resi, mathformat(kires))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)
    resf.write(res)

    # print '{0}'.format(mathformat(finished))
    # print '{0}'.format(mathformat(E0res))
    # print '{0}'.format(mathformat(fsres))
    # print '{0}'.format(mathformat(kres))
    # print '{0}'.format(mathformat(nres))
    # print '{0}'.format(mathformat(n2res))

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

