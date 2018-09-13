#!/usr/bin/env python
"""Main file for DIQT example

Created by Stefano B. Blumberg to illustrate methodology utilised in:
Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images (MICCAI 2018)
"""

import numpy as np
import torch as t
import models

### Illustrate settings from our paper

# Settings used in paper, target is de-shuffled
no_RevNet_layers = 4
no_channels = 6
dtype_np = 'float32'
dtype_pt = t.float32
device = t.device('cuda')
no_RevNet_layers = 250
N,Cin,Hin,Win,Din = 12,6,11,11,11
N,Cout,Hout,Wout,Dout = N,8*Cin,Hin-4,Win-4,Din-4

# Randomly generate input and target
x=np.random.rand(N,Cin,Hin,Win,Din).astype(dtype_np)
y=np.random.rand(N,Cout,Hout,Wout,Dout).astype(dtype_np)
x = t.as_tensor(x,device=device)
y = t.as_tensor(y,device=device)

# ESPCN-RN4 network
DIQT = models.ESPCN_RN(no_RevNet_layers=no_RevNet_layers,
                    no_chans_in=Cin,
                    no_chans_out=Cout,
                    memory_efficient=True)
DIQT.type(dtype_pt)
DIQT=DIQT.to(device=device)

optimizer = t.optim.Adam(DIQT.parameters(), lr=5E-5)

no_epochs = 100
for epoch in range(no_epochs):
    optimizer.zero_grad()

    # Predict target
    y_pred = DIQT(x)

    # Create new leaf node with target
    y_pred = y_pred.detach()
    y_pred.requires_grad_()
    loss = t.mean(t.abs(y_pred-y))

    # Calculate gradient of y_predstefano.blumberg.17@ucl.ac.uk
    loss.backward()

    # Mangual backpropagation
    DIQT.backward(y_pred, y_pred.grad)

    optimizer.step()
    print(epoch)


### Now repeat the above with different settings e.g.
### no_RevNet_layers = 1000
### N,Cin,Hin,Win,Din = 20,20,40,40,40
### More than 6000 convolutional blocks, yet can train on 12GB GPU card!!

print('EOF')
