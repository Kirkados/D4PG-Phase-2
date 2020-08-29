#!/usr/bin/env python

import math, sys, os, numpy as np, time, threading, logging, matplotlib.pyplot as plt
import pdb

def plot_data(data):
    fig = plt.figure(figsize=(15,7))
    t = data[:,0]
    
    # Plotting deep guidance commands
    plt.subplot(511)
    plt.plot(t,data[:,1], '-b', alpha=0.7, label='Quad1 a_X')
    plt.plot(t,data[:,2], '-r', alpha=0.7, label='Quad1 a_Y')
    plt.plot(t,data[:,5], '-g', alpha=0.7, label='Quad1 a_Z')
    plt.plot(t,data[:,3], '--b',  alpha=0.7, label='Quad2 a_X')
    plt.plot(t,data[:,4], '--r',  alpha=0.7, label='Quad2 a_Y')
    plt.plot(t,data[:,6], '--g',  alpha=0.7, label='Quad2 a_Z')
    plt.grid();plt.legend()

    # Plotting positions
    plt.subplot(512)
    plt.plot(t,data[:,7], '-b', alpha=0.7, label='Quad1_X')
    plt.plot(t,data[:,8], '-r', alpha=0.7, label='Quad1_Y')
    plt.plot(t,data[:,11], '--b', alpha=0.7, label='Quad2_X')
    plt.plot(t,data[:,12], '--r', alpha=0.7, label='Quad2_Y')
    plt.grid();plt.legend()

    # Plotting velocities
    plt.subplot(513)
    plt.plot(t,data[:,9],  '--b', alpha=0.7, label='Quad1_Vx')
    plt.plot(t,data[:,10], '--r', alpha=0.7, label='Quad1_Vy')
    plt.plot(t,data[:,13], '-b',  alpha=0.7, label='Quad2_Vx')
    plt.plot(t,data[:,14], '-r',  alpha=0.7, label='Quad2_Vy')
    plt.grid();plt.legend()

    # plotting 3 past actions 
    plt.subplot(514)
    plt.plot(data[:,15:21],label='Quad1_past_actions', alpha=0.3)
    plt.grid()#;plt.legend()
    
    # Plotting runway state
    plt.subplot(515)
    plt.plot(data[:,21:], label='runway_state', alpha = 0.3)
    plt.grid()#;plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    log_filename = sys.argv[1] if len(sys.argv)>1 else '/tmp/log'
    data = np.load(log_filename)
    plot_data(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()