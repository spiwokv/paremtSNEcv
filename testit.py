#!/usr/bin/python

import numpy as np
np.random.seed(666)
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import mdtraj as md

batch_size = 1000
embed_dim = 3
epochs = 10
shuffle_interval = epochs + 1
n_jobs = 1
perplexity = 30.0

def Hbeta(D, beta):
  P = np.exp(-D*beta)
  sumP = np.sum(P)
  H = np.log(sumP)+beta*np.sum(D*P)/sumP
  P = P/sumP
  return H, P

def x2p(X, tol=1e-5, perplexity=30.0):
  print("Computing pairwise distances...")
  (n, d) = X.shape
  sum_X = np.sum(np.square(X), 1)
  D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
  P = np.zeros((n, n))
  beta = np.ones((n, 1))
  logU = np.log(perplexity)
  for i in range(n):
    if i % 500 == 0:
      print("Computing P-values for point %d of %d..." % (i, n))
    betamin = -np.inf
    betamax = np.inf
    Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
    (H, thisP) = Hbeta(Di, beta[i])
    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:
      if Hdiff > 0:
        betamin = beta[i].copy()
        if betamax == np.inf or betamax == -np.inf:
          beta[i] = beta[i] * 2.
        else:
          beta[i] = (beta[i] + betamax) / 2.
      else:
        betamax = beta[i].copy()
        if betamin == np.inf or betamin == -np.inf:
          beta[i] = beta[i] / 2.
        else:
          beta[i] = (beta[i] + betamin) / 2.
      (H, thisP) = Hbeta(Di, beta[i])
      Hdiff = H - logU
      tries += 1
    P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
  print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
  return P

def calculate_P(X):
  print "Computing pairwise distances..."
  n = X.shape[0]
  P = np.zeros([n, batch_size])
  for i in xrange(0, n, batch_size):
    P_batch = x2p(X[i:i + batch_size])
    P_batch[np.isnan(P_batch)] = 0
    P_batch = P_batch + P_batch.T
    P_batch = P_batch / P_batch.sum()
    P_batch = np.maximum(P_batch, 1e-12)
    P[i:i + batch_size] = P_batch
  return P

def KLdivergence(P, Y):
  alpha = embed_dim - 1.
  sum_Y = K.sum(K.square(Y), axis=1)
  eps = K.variable(10e-15)
  D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
  Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
  Q *= K.variable(1 - np.eye(batch_size))
  Q /= K.sum(Q)
  Q = K.maximum(Q, eps)
  C = K.log((P + eps) / (Q + eps))
  C = K.sum(P * C)
  return C

print "Loading data"
refpdb = md.load_pdb("reference.pdb")
traj = md.load("traj_fit.xtc", top="reference.pdb")
traj.superpose(refpdb)
trajsize = traj.xyz.shape
traj2 = np.zeros((trajsize[0], trajsize[1]*3))
for i in range(trajsize[1]):
  traj2[:,3*i]   = traj.xyz[:,i,0]
  traj2[:,3*i+1] = traj.xyz[:,i,1]
  traj2[:,3*i+2] = traj.xyz[:,i,2]
n = traj2.shape[0]
batch_num = int(n // batch_size)
m = batch_num * batch_size

print "Building model"
model = Sequential()
model.add(Dense(500, input_shape=(traj2.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(embed_dim))
model.compile(loss=KLdivergence, optimizer="adam")

print "Training model"
for epoch in range(epochs):
  if epoch % shuffle_interval == 0:
    X = traj2[np.random.permutation(n)[:m]]
    P = calculate_P(X)
  loss = 0.0
  for i in xrange(0, m, batch_size):
    loss += model.train_on_batch(X[i:i+batch_size], P[i:i+batch_size])
  print "Epoch: {}/{}, loss: {}".format(epoch+1, epochs, loss / batch_num)

print "Writing output"
pred = model.predict(traj2)
ofile = open("results.txt", "w")
for i in range(traj2.shape[0]):
  for j in range(embed_dim):
    ofile.write(" %f" % pred[i,j])
  ofile.write("\n")
ofile.close()

