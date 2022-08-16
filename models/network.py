import torch
import torch.nn as nn
from torch.autograd import Variable


def one_hot(x, N):
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1), x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot

def conv(in_planes, out_planes, dilation=1, kernel_size=3, stride=1):
    # p = [(o-1) * s - i + k + (k-1)*(d-1)] / 2.
    # p = [k - 1 + (k-1)*(d-1)] / 2, when i=o, s=1.
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, 
            #dilation=dilation, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
            dilation=dilation, kernel_size=kernel_size, stride=stride, padding=( kernel_size-1+(kernel_size-1)*(dilation-1) )//2),
        nn.GroupNorm(1, out_planes),
        nn.ReLU(inplace=True)
    )

def conv_(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
    )


class CondLayer(nn.Module):
    """
    pixel-wise feature modulation.
    """
    def __init__(self, n_channel=None):
        super(CondLayer, self).__init__()
        self.n_channel = n_channel
        self.elu = nn.ELU(inplace=True)
        self.ReLU = nn.ReLU(inplace=True)
        if not self.n_channel is None: self.GN = nn.GroupNorm(1, self.n_channel)
    def forward(self, x, gammas, betas):
        if not self.n_channel is None: return self.ReLU(self.GN((gammas * x) + betas))
        else: return self.ReLU((gammas * x) + betas)




class Classifier(nn.Module):
    """
    Scene Region Classification. 
    """
    def __init__(self, 
        n_class, 
        training=True, 
        #dataset='7S'
        ):

        super(Classifier, self).__init__()
        self.training = training
        #self.dataset = dataset
        self.n_class = n_class
        
        #n_channels = [256, 512, 1024]
        n_channels = [256, 256, 512] # the light version.
        n_channels_c = [512, 256, self.n_class]
        n_channels_g = [256, 512]

        # level 1 base network layers.
        self.conv4a_l1 = conv(n_channels[0], n_channels[1], kernel_size=5, dilation=5)
        self.conv4b_l1 = conv(n_channels[1], n_channels[2], kernel_size=5, dilation=5) 
        self.convc1_l1 = conv(n_channels[2], n_channels_c[0], kernel_size=1) 
        self.convc6 = conv(n_channels_c[0], n_channels_c[1], kernel_size=1)
        self.convoutc1 = conv_(n_channels_c[1], n_channels_c[2])

        # level 2 hyper network layers.
        self.cond = CondLayer(n_channels_c[1])
        self.gconv1_1 = conv(n_channels_c[-1], n_channels_g[0], kernel_size=1)
        self.gconv1_4 = conv(n_channels_g[0], n_channels_g[1], kernel_size=1)
        self.gconv1_gamma_2 = conv(n_channels_g[0], n_channels_c[1], kernel_size=1)
        self.gconv1_beta_2 = conv(n_channels_g[0], n_channels_c[1], kernel_size=1)

        # level 2 base network layers.
        self.conv4a_l2 = conv(n_channels[0], n_channels[1], kernel_size=3, dilation=1) 
        self.conv4b_l2 = conv(n_channels[1], n_channels[2], kernel_size=3, dilation=1)
        self.convc1_l2 = conv(n_channels[2], n_channels_c[0], kernel_size=1) 
        self.convc3 = conv(n_channels_c[0], n_channels_c[1], kernel_size=1)
        self.convoutc2 = conv_(n_channels_c[1], n_channels_c[2])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, lbl_1=None, lbl_2=None):
        #######################
        # level 1 base network.
        #######################
        # feature pattern.
        x1 = self.conv4a_l1(x)
        x1 = self.conv4b_l1(x1)
        # classifier.
        c1 = self.convc1_l1(x1)
        c1 = self.convc6(c1)
        # output the classification probability.
        out_lbl_1 = self.convoutc1(c1)

        ########################
        # level 2 hyper network.
        ########################
        if self.training is not True:
            lbl_1 = torch.argmax(out_lbl_1, dim=1)
            lbl_1 = one_hot(lbl_1, out_lbl_1.size()[1]) 
        ebd = self.gconv1_1(lbl_1)
        out_gconv1_gamma_2 = self.gconv1_gamma_2(ebd)   
        out_gconv1_beta_2 = self.gconv1_beta_2(ebd)     

        #######################
        # level 2 base network.
        #######################
        # feature pattern.
        x2 = self.conv4a_l2(x) 
        x2 = self.conv4b_l2(x2)
        # classifier. 
        c2 = self.convc1_l2(x2)
        c2 = self.cond(self.convc3(c2), out_gconv1_gamma_2, out_gconv1_beta_2) # modulation.
        # output the classification probability.
        out_lbl_2 = self.convoutc2(c2)

        if self.training is not True:
            lbl_2 = torch.argmax(out_lbl_2, dim=1)
            lbl_2 = one_hot(lbl_2, out_lbl_2.size()[1])

        return out_lbl_2, out_lbl_1 # each of [BS=1, self.n_class, 60, 80].


    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def clone(self):
        clone = Classifier(n_class=self.n_class, training=self.training)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda(): clone.cuda()
        return clone

    def point_grad_to(self, target):
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_() # not sure this is required?
            p.grad.data.add_(p.data - target_p.data)

