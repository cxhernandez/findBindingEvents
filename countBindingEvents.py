#!/usr/bin/env python
# cython: profile=True
# -*- coding: utf-8 -*-
###############################################################################
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
###############################################################################
"""


This script uses hybrid MPI/OpenMP paralleism in addition to highly optimized
SIMD vectorization within the compute kernels. Using multiple MPI processes
requires running this command using your MPI implementation's process manager,
e.g. `mpirun`, `mpiexec`, or `aprun`. The number of OpenMP threads can be
controled by setting the OMP_NUM_THREADS environment variable. (e.g.
$ export OMP_NUM_THREADS=4; mpirun -np 16 python tent.py <options>)

Authors: Carlos Xavier Hernandez
"""
#-----------------------------------
# Imports
#-----------------------------------
from __future__ import print_function
import glob, argparse, os, sys, time, datetime, itertools, warnings
import numpy as np
try:
    import mdtraj as md
except ImportError:
    print("This package requires the latest development version of MDTraj")
    print("which can be downloaded from https://github.com/rmcgibbo/mdtraj")
    sys.exit(1)
try:
    from mpi4py import MPI
except:
    print("This package requires mpi4py, which can be downloaded")
    print("from https://pypi.python.org/pypi/mpi4py")
    sys.exit(1)
try:
    import pymc as pm
except ImportError:
    print("This package requires pymc, which can be downloaded")
    print("from https://pypi.python.org/pypi/pymc")
    sys.exit(1)

#-----------------------------------
# Globals
#-----------------------------------
COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size

def rmsd(traj, ref, idx):
    return np.sqrt(np.sum(np.square(traj[:,idx,:] - ref[:,idx,:]),axis=(1,2))/idx.shape[0])

def printM(message, *args):
    if RANK == 0:
        if len(args) == 0:
            print(message)
        else:
            print(message % args)
            
class timing(object):
    "Context manager for printing performance"
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, ty, val, tb):
        end = time.time()
        print("<RANK %d> PERFORMANCE [%s] : %0.3f seconds" % (RANK, self.name, end-self.start))
        return False
            
def parse_cmdln():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-td', '--dir', dest='dir',help='Directory containing trajectories')
    parser.add_argument('-ext', '--ext', dest='ext', help='File extension', default='dcd')
    parser.add_argument('-ref', '--ref', dest='reference', help='Reference pdb of bound structure')
    parser.add_argument('-s', '--stride', dest='stride', help='Stride', default=10)
    parser.add_argument('-p', '--protein', dest='prot', help='Protein indices', default=None)
    parser.add_argument('-l', '--ligand', dest='lig', help='Ligand indices', default=None)
    parser.add_argument('-d', '--cutoff', dest='d', help='RMSD cutoff', default=0.5)
    parser.add_argument('-c', '--significance', dest='c', help='Signficance cutoff', default=5.0)
    #parser.add_argument('-i', '--idx', dest='idx', help='Residues to compare in bound pose', default=None)
    args = parser.parse_args()
    return args
    

def init_gamma_parms(r):
    a = 1/np.mean(r)
    return a

def findEvent(metric, steps = 10000, burn=0.1, thin=1):
    a = init_gamma_parms(metric)
    mu1, mu2, tau = pm.Exponential('u1',a), pm.Exponential('u2',a), pm.DiscreteUniform("tau", lower=0, upper=metric.shape[0])
    
    @pm.deterministic
    def params(a1=mu1, a2=mu2,tau=tau):
        out = np.zeros(metric.shape[0])
        out[:tau] = a1  
        out[tau:] = a2 
        return out
    
    obs = pm.Poisson("obs",params,value=metric,observed=True)
    model = pm.Model([obs,mu1,mu2,tau])
    
    mcmc = pm.MCMC(model)
    mcmc.sample(steps, int(np.rint(burn*steps)), thin, progress_bar=False)
    
    mu1_samples = mcmc.trace('u1')[:]
    mu2_samples = mcmc.trace('u2')[:]
    tau_samples = mcmc.trace('tau')[:]
    
    sig = (np.mean(mu1_samples)-np.mean(mu2_samples))/np.sqrt(np.var(mu1_samples)+np.var(mu2_samples) + 1E-5)
    
    return sig, np.median(mu1_samples), np.median(mu2_samples)
    
def create_features(ref, prot, lig, d):
    contacts = md.compute_contacts(ref,contacts=list(itertools.product([ref.topology.atom(i).residue.index for i in prot],[ref.topology.atom(i).residue.index for i in lig]))
    atom_set = contacts[1][np.where(contacts[0]<d),:]
    return atom_set
   
def calculate_metrics(traj, features, d):
    contacts = md.compute_contacts(traj, contacts = features)
    h = np.sum(contacts[0] < .5, axis=1)
    return h
    
def main(trajectories, topology, prot, lig, idx, stride, d, c):
    bind = unbind = 0
    ref = topology.atom_slice(atom_indices = idx, inplace=False)
    features = create_features(ref, prot, lig, d)
    for trajectory in trajectories:
        with timing('Finding binding events...'):
            traj = md.load(trajectory, top = topology, stride = stride, atom_indices = idx)
            traj.superpose(ref, atom_indices = prot)
            h, m1, m2  =  create_metrics(traj, features, d)
            q = findEvent(h)
            if (c < q)*(m2 >= features.shape)*(m1 < features.shape):
                bind += 1
            elif (-c > q)*(m1 >= features.shape)*(m2 < features.shape):
                unbind += 1
    
    COMM.Barrier()
    n_bind = n_unbind = 0
    COMM.reduce(bind, n_bind, MPI.SUM)
    COMM.reduce(unbind, n_unbind, MPI.SUM)
    
    printM(u'Found %s binding events and %s unbinding events (sigma is %s)' % (n_bind,n_unbind,c))
            
    
if __name__ == "__main__":
    
    options = parse_cmdln()
    topology = md.load(options.reference)
    
    if RANK == 0:
        trajectories = glob.glob(options.dir + "/*." + options.ext)
        try:
            if not options.dir:
                parser.error('Please supply a directory.')
            if not options.reference:
                parser.error('Please supply a reference file.')
            if not trajectories:
                print("No trajectories found.")
                sys.exit(1)
            if len(trajectories) < SIZE:
                print("There are more nodes than trajectories.")
                sys.exit(1)
        except SystemExit:
            if SIZE > 1:
                COMM.Abort()
            exit()
        trajectories = [trajectories[i::SIZE] for i in range(SIZE)]
        prot = np.loadtxt(options.prot, dtype=int)
        lig = np.loadtxt(options.lig, dtype=int)
        idx = np.hstack((prot,lig))
        prot = np.arange(0,len(prot))
        lig = np.arange(len(prot),len(prot)+len(lig))
    else:
        trajectories = lig = idx = prot = None
        
    
    trajectories = COMM.scatter(trajectories, root=0)
    prot = COMM.bcast(prot, root=0)
    lig = COMM.bcast(lig, root=0)
    idx = COMM.bcast(idx, root=0)
    
    printM('Starting...')
    
    main(trajectories, topology, prot, lig, idx, int(options.stride), options.d, options.c)
