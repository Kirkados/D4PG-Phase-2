#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import glob


time_to_solves = []
fig = plt.figure(figsize=(15,7))
plt.gca().set_aspect("equal")


# Generate grid points
width_elements = 8
length_elements = 4
# Draw runway
#boundary_points = np.array([[-0.4,-0.4],[0.4,-0.4],[0.4,0.4],[-0.4,0.4],[-0.4,-0.4]])
boundary_points = np.array([[-62,-12.5/2],[62,-12.5/2],[62,12.5/2],[-62,12.5/2],[-62,-12.5/2]])
plt.plot(boundary_points[:,0],boundary_points[:,1], 'k')

# Plot grid elements
for i in range(width_elements):
    plt.plot([-62+62*2/width_elements*i,-62+62*2/width_elements*i],[-12.5/2,12.5/2]) 
    
for i in range(length_elements):
    plt.plot([-62,62],[-12.5/2+12.5/2*2/length_elements*i,-12.5/2+12.5/2*2/length_elements*i]) 
    
    
for file in glob.glob("*2q_f2*.txt"):        
    
    log_filename = file
    data = np.load(log_filename)
    
    t = data[:,0]
    plt.plot(data[:,7],data[:,8],  '-bo', alpha=0.2, label='quad1')
    #plt.plot(data[:,11],data[:,12],  '-ro', alpha=0.2, label='quad2') # OBERVATION
    plt.plot(data[:,13],data[:,14],  '-ro', alpha=0.2, label='quad2') # TOTAL_STATE
    
    plt.figure()
    plt.gca().set_aspect("equal")
    plt.plot(boundary_points[:,0],boundary_points[:,1], 'k')
    # Plot grid elements
    for i in range(width_elements):
        plt.plot([-62+62*2/width_elements*i,-62+62*2/width_elements*i],[-12.5/2,12.5/2]) 
        
    for i in range(length_elements):
        plt.plot([-62,62],[-12.5/2+12.5/2*2/length_elements*i,-12.5/2+12.5/2*2/length_elements*i]) 
    plt.quiver(data[:,7],data[:,8],data[:,1],data[:,2],scale=250)
    plt.quiver(data[:,13],data[:,14],data[:,3],data[:,4],scale=250)
    
    plt.tight_layout()
    plt.show()
    
    print("This flight took: %.1f seconds" %t[-1])
    if t[-1] < 2000.52:
        time_to_solves.append(t[-1])
        
    # Plotting the velocity and acceleration curve to look for delays
    plt.figure()
    plt.plot(t, data[:,16], '-bo', label='Vx')
    plt.plot(t, data[:,3], '-ko', label='Ax_command')
    plt.plot(t, data[:,17], '-ro', label='Vy')
    plt.plot(t, data[:,4], '-go', label='Ay_command')
    plt.plot(t, data[:,15], label='altitude')
    plt.legend()
    plt.grid()
    
    #print(np.max(np.diff(t)))
    #print(np.diff(data[:,7]))

print("Average time: %.1f s; best time %.1f s; worst time %.1f s, for %i trials" %(np.average(np.asarray(time_to_solves)), np.min(time_to_solves), np.max(time_to_solves), len(time_to_solves)))