"""Module containing, RevNet and DIQT Network

Created by Stefano B. Blumberg to illustrate methodology utilised in:
Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images (MICCAI 2018)
"""

import torch as t ; import torch.nn as nn


class ESPCN_RN(nn.Module):
    """ESPCN_RN-N (N:=no_RevNet_layers), from
    Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images

    We have omitted the final shuffle for simplicity.

    backpropagation is performed manually.  Please see Main file on how to integrate this
    """

    def __init__(self,
                no_RevNet_layers,
                no_chans_in=6,
                no_chans_out=48,
                memory_efficient=True
                ):
        """
        Args:
            no_RevNet_layers (int): Number of RevNet layers per stack
            no_chans_in (int): Number of input channels
            memory_efficient (bool): Use memory-efficient technique
        """

        super().__init__()

        noChansin0,noChansout0 = no_chans_in,50
        self.rn0 = RevNet(noChansin0//2,
                        noChansin0//2,
                        no_RevNet_layers=no_RevNet_layers,
                        memory_efficient=memory_efficient)
        self.conv0 = nn.Sequential(nn.Conv3d(noChansin0, noChansout0, kernel_size=3, padding=0),
                                nn.ReLU())

        noChansin1,noChansout1 = 50,100
        self.rn1 = RevNet(noChansin1//2,
                        noChansin1//2,
                        no_RevNet_layers=no_RevNet_layers,
                        memory_efficient=memory_efficient)
        self.conv1 = nn.Sequential(nn.Conv3d(noChansin1, noChansout1, kernel_size=1, padding=0),
                                nn.ReLU())

        noChansin2,noChansout2 = 100,no_chans_out
        self.rn2 = RevNet(noChansin2//2,
                        noChansin2//2,
                        no_RevNet_layers=no_RevNet_layers,
                        memory_efficient=memory_efficient)
        self.conv2 = nn.Sequential(nn.Conv3d(noChansin2, noChansout2, kernel_size=3, padding=0))

    def forward(self, X):
        """Memory-efficient forward pass. Cache intermediate activations as attributes"""

        self.inpConv0 = self.rn0(X)
        X = self.conv0(self.inpConv0)
        self.inpConv1 = self.rn1(X)
        X = self.conv1(self.inpConv1)
        self.inpConv2 = self.rn2(X)
        X = self.conv2(self.inpConv2)
        return X

    def backward(self,Y,YGrad):
        """Memory-efficient backward pass, computes gradients"""

        conv2_root = self.inpConv2.requires_grad_()
        Y =  self.conv2(conv2_root)
        t.autograd.backward([Y], [YGrad])
        _,YGrad = self.rn2.backward(self.inpConv2, conv2_root.grad)

        conv1_root = self.inpConv1.requires_grad_()
        Y =  self.conv1(conv1_root)
        t.autograd.backward([Y],[YGrad])
        _,YGrad = self.rn1.backward(self.inpConv1,conv1_root.grad)

        conv0_root = self.inpConv0.requires_grad_()
        Y =  self.conv0(conv0_root)
        t.autograd.backward([Y],[YGrad])
        _,_ = self.rn0.backward(self.inpConv0,conv0_root.grad)



class RevNet(nn.Module):
    """RevNet Class"""

    def __init__(self,
                noChans1=3,
                noChans2=3,
                no_RevNet_layers=10,
                whole=True,
                memory_efficient=True):
        super().__init__()
        self.noChans1 = noChans1
        self.no_RevNet_layers = no_RevNet_layers
        self.whole = whole
        self.memory_efficient=memory_efficient

        self.CreateFG(noChans1, noChans2, no_RevNet_layers)

    def CreateFG(self, noChans1, noChans2, no_RevNet_layers):
        """Construct residual functions, split input channels into noChans1 + noChans2

         e.g. F5 is residual function F for layer 5
         """

        for layer in range(no_RevNet_layers):
            """
            # An example of another (basic) residual block
            Basic = nn.Sequential(nn.BatchNorm3d(noChans2),
                                nn.ReLU(),
                                nn.Conv3d(noChans2, noChans2, kernel_size=3, padding=1),
                                nn.BatchNorm3d(noChans2),
                                nn.ReLU(),
                                nn.Conv3d(noChans2, noChans1, kernel_size=3, padding=1))
            """
            residual_1 =  nn.Sequential(nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans1, kernel_size=1, padding=0),
                                    nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans1, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans2, kernel_size=1, padding=0))

            residual_2 =  nn.Sequential(nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans1, kernel_size=1, padding=0),
                                    nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans1, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(noChans1),
                                    nn.ReLU(),
                                    nn.Conv3d(noChans1, noChans2, kernel_size=1, padding=0))

            setattr(self,'F'+str(layer), residual_1)
            setattr(self,'G'+str(layer), residual_2)

    def ForwardPassLayer(self,
                        layer_no,
                        X):
        """ Forward Pass of a RevNet layer

        Args:
             (x1,x2) tuple of tensors to layer layer_no
        Returns:
            Forward pass of layer layer_no as tuple tensors
        """

        x1,x2 = X
        z1 = x1 + getattr(self, 'F'+str(layer_no))(x2)
        y2 = x2 + getattr(self, 'G'+str(layer_no))(z1)
        y1 = z1
        return (y1,y2)

    def forward(self, X):
        """Forward pass of multiple reversilble layers in a block

        Args:
            (self.whole bool): T means we have to split the input, F it is passed as a tuple
        """

        if self.whole:
            X = (X[:,0:self.noChans1,...].contiguous(), X[:,self.noChans1:,...].contiguous())
        for layer_no in range(self.no_RevNet_layers):
            # Least memory saving option
            if self.memory_efficient:
                with t.no_grad():
                    Y = self.ForwardPassLayer(layer_no,X)
                    X = Y
            # Forward pass agnostic if self.memory_efficient=False
            if not self.memory_efficient:
                Y = self.ForwardPassLayer(layer_no,X)
                X = Y
            del Y ; t.cuda.empty_cache()
        X = t.cat((X[0],X[1]),1)
        return X

    def BackwardsPassLayer(self,
                            layer_no,
                            Y,
                            YHat):
        """Memory-efficient backwards pass for a RevNet layer

        Assume self.memory_efficient=True
        Args:
            layer_no (int): RevNet layer no. in stack
            Y = (y1,y2) (2-tensor-tuple): Output of Revnet layer_no
            YHat = (y1Hat, y2Hat) (2-tensor-tuple): Gradients of Revnet layer_no output
        """

        y1,y2 = Y
        y1Hat,y2Hat = YHat
        with t.no_grad():
            z1 = y1 # DO need to make a copy here? id(z1)=id(y1)
            x2 = y2 - getattr(self, 'G'+str(layer_no))( z1 )
            x1 = z1 - getattr(self, 'F'+str(layer_no))( x2 )
        z1.requires_grad_()
        y2Part = getattr(self, 'G'+str(layer_no))(z1)
        t.autograd.backward([y2Part],[y2Hat])
        z1.grad += y1Hat
        x2.requires_grad_()
        z1Part = getattr(self, 'F'+str(layer_no))(x2)
        t.autograd.backward([z1Part],[z1.grad])
        x2.grad += y2Hat
        del y1,y2,Y,y2Part
        t.cuda.empty_cache()
        return ((x1,x2),(z1.grad,x2.grad))

    def backward(self,
                Y,
                YHat):
        """Memory-efficient backwards pass for a RevNet layers in a stack

        Assume self.memory_efficient=True
        Args:
            Y (2-tensor-tuple or tensor): Revnet stack output, format depends on self.whole
            YHat (2-tensor-tuple or tensor): Y gradients, format depends on self.whole
        """

        if not self.memory_efficient: Exception('Using Manual backpropagation when shouldnt')

        if self.whole:
            Y = (Y[:,0:self.noChans1,...].contiguous(), Y[:,self.noChans1:,...].contiguous())
            YHat = (YHat[:,0:self.noChans1,...].contiguous(),
                    YHat[:,self.noChans1:,...].contiguous())
        Y = Y[0].data, Y[1].data
        for layer_no in reversed(range(self.no_RevNet_layers)):
            (Y,YHat) = self.BackwardsPassLayer(layer_no,Y,YHat)
            print(layer_no)
        Y = t.cat( (Y[0],Y[1]),1)
        YHat = t.cat((YHat[0],YHat[1]),1)
        return (Y, YHat)
