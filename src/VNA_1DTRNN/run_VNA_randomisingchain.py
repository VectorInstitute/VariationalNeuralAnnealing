#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------------------------------------
"""
This implementation is based on RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Title : Implementation of Variational Neural Annealing for the random Ising chain with open boundary conditions
Description : This code uses 1D Tensorized RNN cells to run variational neural annealing both classical
and quantum on the random ising chain.
"""
#-----------------------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import time
import random

from RNNWavefunction import RNNWavefunction
from TensorizedRNNCell import TensorizedRNNCell
from Helper_functions import *

#Seeding for reproducibility purposes
seed = 111
tf.compat.v1.reset_default_graph()
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

#### Hyperparams
# Note:
# If Bx0=0, then this code will run Variational Classical Annealing (VCA).
# If T0=0, then this code will run Variational Quantum Annealing (VQA).
# If both are zero, then this algorithm will correspond to classical quantum optimization (CQO).
# For more details, please check Ref. https://arxiv.org/abs/2101.10154.
N = 20 #number of spins in the chain
num_units = 20 #number of memory units
numlayers = 1 #number of layers
numsamples = 50 #Batch size
lr = 1e-3 #learning rate
T0 = 2 #Initial temperature
Bx0 = 2 #initial magnetic field
num_warmup_steps = 1000 #number of warmup steps
num_annealing_steps = 500 #number of annealing steps
num_equilibrium_steps = 5 #number of training steps after each annealing step
activation_function = tf.nn.elu #activation of the RNN cell

Jz = (2*np.random.randint(0,high=2,size=N-1)-1) #Random couplings of the random Ising chain taken from {-1,1} - dependent on the chosen seed

units=[num_units]*numlayers #list containing the number of hidden units for each layer of the RNN

print('\n')
print("Number of spins =", N)
print("Initial_temperature =", T0)
print('Seed = ', seed)

num_steps = num_annealing_steps*num_equilibrium_steps + num_warmup_steps #Total number of gradient steps

print("\nNumber of annealing steps = {0}".format(num_annealing_steps))
print("Number of training steps = {0}".format(num_steps))
print("Number of layers = {0}\n".format(numlayers))


# Intitializing the RNN-----------
RNNWF = RNNWavefunction(N,units=units,cell=TensorizedRNNCell, activation = activation_function, seed = seed) #contains the graph with the RNNs

#Building the graph -------------------
with tf.compat.v1.variable_scope(RNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with RNNWF.graph.as_default():

        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
        learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)

        #Defining the optimizer
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

        #Defining Tensorflow placeholders
        Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
        sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,N])
        log_probs_forgrad = RNNWF.log_probability(sampleplaceholder_forgrad,inputdim=2)

        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,N))
        log_probs_tensor=RNNWF.log_probability(samples_placeholder,inputdim=2)
        samplesandprobs = RNNWF.sample(numsamples=numsamples,inputdim=2)

        T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

        #Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
        Floc = Eloc + T_placeholder*log_probs_forgrad
        cost = tf.reduce_mean(tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

        gradients, variables = zip(*optimizer.compute_gradients(cost))
        #Calculate Gradients---------------

        #Define the optimization step
        optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)

        #Tensorflow saver to checkpoint
        saver=tf.compat.v1.train.Saver()

        #For initialization
        init=tf.compat.v1.global_variables_initializer()
        initialize_parameters = tf.initialize_all_variables()
#----------------------------------------------------------------

#Starting Session------------
#GPU management
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

#Start session
sess=tf.compat.v1.Session(graph=RNNWF.graph, config=config)
sess.run(init)

#Loading previous trainings----------
    ### To be implemented
#------------------------------------

## Run Variational Annealing
with tf.compat.v1.variable_scope(RNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with RNNWF.graph.as_default():

      #To store data
      meanEnergy=[]
      varEnergy=[]
      varFreeEnergy = []
      meanFreeEnergy = []
      samples = np.ones((numsamples, N), dtype=np.int32)
      queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
      log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

      T = T0 #initializing temperature
      Bx = Bx0 #initializing magnetic field

      sess.run(initialize_parameters) #Reinitialize the parameters

      start = time.time()

      for it in range(len(meanEnergy),num_steps+1):
            #Annealing
            if it>=num_warmup_steps and  it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it % num_equilibrium_steps == 0:
              annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
              T = T0*(1-annealing_step/num_annealing_steps)
              Bx = Bx0*(1-annealing_step/num_annealing_steps)

            #Showing current status after that the annealing starts
            if it%num_equilibrium_steps==0:
              if it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it>=num_warmup_steps:
                  annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                  print("\nAnnealing step: {0}/{1}".format(annealing_step,num_annealing_steps))

            #Getting samples and log_probs from the RNN
            samples, log_probabilities = sess.run(samplesandprobs)

            # Estimating the local energies
            local_energies = Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess)

            #computing <H> and var(<H>)
            meanE = np.mean(local_energies)
            varE = np.var(local_energies)

            #adding elements to be saved
            meanEnergy.append(meanE)
            varEnergy.append(varE)

            #computing F and var(F)
            meanF = np.mean(local_energies+T*log_probabilities)
            varF = np.var(local_energies+T*log_probabilities)

            #adding elements to be saved
            meanFreeEnergy.append(meanF)
            varFreeEnergy.append(varF)

            #Printing progress
            if it%num_equilibrium_steps==0:
                print('mean(E): {0}, mean(F): {1}, var(E): {2}, var(F): {3}, #samples {4}, #Training step {5}'.format(meanE,meanF,varE,varF,numsamples, it))
                print("Temperature: ", T)
                print("Magnetic field: ", Bx)

            #Here we produce samples at the end of annealing
            if it == num_annealing_steps*num_equilibrium_steps + num_warmup_steps:

                numsamples_estimation = 10**5 #Num samples to be obtained at the end
                Nsteps = 20 #The number of steps taken to get "numsamples_estimation" samples (to avoid memory allocation issues)
                numsamples_perstep = numsamples_estimation//Nsteps

                samplesandprobs_final = RNNWF.sample(numsamples=numsamples_perstep,inputdim=2)
                energies = np.zeros((numsamples_estimation))
                solutions = np.zeros((numsamples_estimation, N))
                print("\nSaving energy and variance before the end of annealing")

                for i in range(Nsteps):
                    # print("\nsampling started")
                    samples_final, _ = sess.run(samplesandprobs_final)
                    # print("\nsampling finished")
                    energies[i*numsamples_perstep:(i+1)*numsamples_perstep] = Ising_diagonal_matrixelements(Jz,samples_final)
                    solutions[i*numsamples_perstep:(i+1)*numsamples_perstep] = samples_final
                    print("Sampling step:" , i+1, "/", Nsteps)

                print("meanE = ", np.mean(energies))
                print("varE = ", np.var(energies))
                print("minE = ",np.min(energies))

            #Run gradient descent step
            sess.run(optstep,feed_dict={Eloc:local_energies, sampleplaceholder_forgrad: samples, learningrate_placeholder: lr, T_placeholder:T})

            if it%5 == 0:
                print("Elapsed time is =", time.time()-start, " seconds")
                print('\n\n')

#----------------------------
