#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import glob


time_to_solves = []
fig = plt.figure(figsize=(15,7))
plt.gca().set_aspect("equal")

# Draw runway
boundary_points = np.array([[-0.4,-0.4],[0.4,-0.4],[0.4,0.4],[-0.4,0.4],[-0.4,-0.4]])
plt.plot(boundary_points[:,0],boundary_points[:,1], 'k')
for file in glob.glob("*.txt"):        
    
    log_filename = file
    data = np.load(log_filename)
    
    t = data[:,0]
    plt.plot(-data[:,7],data[:,8],  '-b', alpha=0.2, label='quad1')
    plt.plot(-data[:,11],data[:,12],  '-r', alpha=0.2, label='quad2')
    
    plt.tight_layout()
    plt.show()
    
    print("This flight took: %.1f seconds" %t[-1])
    time_to_solves.append(t[-1])
    
    print(np.max(np.diff(t)))
    print(np.diff(data[:,7]))

print("Average time: %.1f s; best time %.1f s; worst time %.1f s, for %i trials" %(np.average(np.asarray(time_to_solves)), np.min(time_to_solves), np.max(time_to_solves), len(time_to_solves)))