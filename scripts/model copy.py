import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import qr, svd, solve, eigh
from scipy.signal import butter, sosfiltfilt

class ChannelOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, optimizer='trca'):
        super(ChannelOptimizer, self).__init__()
        self.Ci = None # input channel number
        self.Co = num_out 
        self.K  = None # class number 
        self.T  = None # sample points
        self.S  = None # spatial filter: numerator
        self.Q  = None # spatial filter: denominator
        self.W  = None # spatial filter
        self.G  = None # average of grand average MRCP as bias
        self.M  = None # grand average MRCP
        self.optimizer = optimizer

    def fit(self, X:np.ndarray, y:np.ndarray):
        # X: trial, channel, time
        # y: trial
        y = y.ravel()
        _, self.Ci,  self.T = X.shape
        category = sorted(np.unique(y))
        self.K = len(category)
        self.M = np.zeros((self.K, self.Ci, self.T))
        optimizer = self.optimizer
        if optimizer is None or optimizer.lower() == 'none':
            self.W = np.eye(self.Ci)#[:, self.Ci//2-self.Co//2:self.Ci//2+1+self.Co//2]
            for i, c in enumerate(category):
                x = X[y==c]
                self.M[i] = x.mean(0)
            self.G = self.M.mean(0)
        elif optimizer == 'trca':
            self.S = np.zeros((self.K, self.Ci, self.Ci))
            self.Q = np.zeros((self.K, self.Ci, self.Ci))
            for i, c in enumerate(category):
                x = X[y==c]
                self.S[i], self.Q[i]=self.trca(x)
                self.M[i] = x.mean(0)
            _, self.W = eigh(self.S.mean(0), self.Q.mean(0), driver='gv', check_finite=False)
            self.W = self.W[:,::-1]
            # self.W = self.geigensolve(self.S.mean(0), self.Q.mean(0))
            self.W = self.W[:self.Ci, :self.Co]
            self.G = self.M.mean(0)
        elif optimizer == 'tdlda':
            self.G = X.mean(0)
            self.S = np.zeros((self.K, self.Ci, self.Ci))
            self.Q = np.zeros((self.K, self.Ci, self.Ci))
            for i, c in enumerate(category):
                x = X[y==c]
                self.M[i] = x.mean(0)
                self.S[i] = (self.M[i]-self.G)@(self.M[i]-self.G).T
                for xx in x:
                    self.Q[i] += (xx-self.M[i])@(xx-self.M[i]).T
            _, self.W = eigh(self.S.mean(0), self.Q.mean(0), driver='gv', check_finite=False)

            # W is sorted in the ascending order now, we need to reverse it
            self.W = self.W[:,::-1]
            # self.W = self.geigensolve(self.S.mean(0), self.Q.mean(0))
            self.W = self.W[:self.Ci, :self.Co]
        return self

    def transform(self, X:np.ndarray):
        # X: trial, channel, time
        W = self.W
        if len(X.shape)==2:
            X = np.expand_dims(X, 0)
        newX = np.einsum('ij,kjl->kil',W.T,X)
        return np.expand_dims(newX, 1)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    @staticmethod
    def trca(data:np.ndarray):
        # trial, channel, time
        n_l, n_c, n_t  = data.shape
        UX = np.swapaxes(data, 0, 1).reshape((n_c, n_l*n_t))
        UX = UX - np.mean(UX, axis=1, keepdims=True)
        Q = UX@UX.T/(n_t*n_l)

        data -= np.mean(data, axis=2, keepdims=True)
        U = data.sum(0)
        V = np.zeros((n_c, n_c))
        for k in range(n_l):
            V += data[k]@data[k].T
        S = (U@U.T-V)/(n_t*n_l*(n_l-1))
        return S, Q
    
    @staticmethod
    def geigensolve(S, Q):
        [d,v] =np.linalg.eig(Q)
        d=np.real(d)
        index=np.argsort(-d)
        d=d[index]
        v=v[:,index]
        P=np.diag(1.0/np.sqrt(d))*v.T
        T=P@S@P.T
        [d,v] =np.linalg.eig(T)
        d=np.real(d)
        index=np.argsort(-np.abs(d))
        v=v[:,index]
        w=P.T@v
        W=w/np.linalg.norm(w,2,axis=0,keepdims=True)
        return W
    

class STRCA(ChannelOptimizer, BaseEstimator, TransformerMixin):
    def __init__(self, Co, optimizer, ifG):
        super(STRCA, self).__init__(Co, optimizer)
        self.optimizer = optimizer
        self.ifG = ifG

    def fit(self, X:np.ndarray, y:np.ndarray):
        super(STRCA, self).fit(X, y)

    def transform(self, X:np.ndarray, ifG=True):
        # X: trial, channel, time
        W = self.W
        if self.ifG:
            G = self.G
        else:
            G = np.zeros_like(self.G)
        if len(X.shape)==2:
            X = np.expand_dims(X, 0)
        pattern = np.zeros((X.shape[0], self.K, 8))
        for i in range(X.shape[0]):
            for c in range(self.K):
                x  = X[i,:,:]-G
                mx = self.M[c,:,:]-G
                mx_= (self.M.sum(0)-self.M[c,:,:])/(self.K-1)-G
                pattern[i,c,:]=self.corrfea(x.T, mx.T, mx_.T, W)
        return pattern.reshape((X.shape[0], -1))
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    @staticmethod
    def canoncorr(x:np.ndarray, y:np.ndarray):
        n, p1 = x.shape
        _, p2 = y.shape

        [Q1, R1, perm1] = qr(x, mode='economic', pivoting=True)
        rankX = np.sum(np.abs(np.diag(R1))>np.finfo(R1.dtype).eps*max(n,p1))
        if rankX < p1:
            Q1 = Q1[:,:rankX]
            R1 = R1[:rankX, :rankX]
        
        [Q2, R2, perm2] = qr(y, mode='economic', pivoting=True)
        rankY = np.sum(np.abs(np.diag(R2))>np.finfo(R2.dtype).eps*n*p2)
        if rankY < p2:
            Q2 = Q2[:,:rankY]
            R2 = R2[:rankY, :rankY]
        d = min(rankX, rankY)
        [U, S, Vh] = svd(Q1.T@Q2, full_matrices=False, compute_uv=True, lapack_driver='gesvd')
        A = solve(R1, U[:,:d]*np.sqrt(n-1))
        B = solve(R2, Vh[:d,:].T*np.sqrt(n-1))
        A[perm1, :] = np.concatenate((A, np.zeros((p1-rankX, d))), axis=0)
        B[perm2, :] = np.concatenate((B, np.zeros((p2-rankY, d))), axis=0)

        return A, B
    
    def corrfea(self, x:np.ndarray, mx:np.ndarray, mx_:np.ndarray, W:np.ndarray):
        # x  : time, channel
        # mx : time, channel, target class
        # mx_: time, channel, mean of other classes
        p = np.zeros((2,4))
        p[0, 0] = self.pearcorr2d(x, mx)
        p[0, 1] = self.pearcorr2d(x@W, mx@W) ### 1
        # A, B    = self.canoncorr(x, mx)
        # p[0, 2] = self.pearcorr2d(x@A, mx@A)
        # p[0, 3] = self.pearcorr2d(x@B, mx@B) ### 3
        
        p[1, 0] = self.pearcorr2d(x-mx_, mx-mx_)
        p[1, 1] = self.pearcorr2d((x-mx_)@W, (mx-mx_)@W)
        # A, B    = self.canoncorr(x-mx_, mx-mx_)
        # p[1, 2] = self.pearcorr2d((x-mx_)@A, (mx-mx_)@A) ### 6
        # p[1, 3] = self.pearcorr2d((x-mx_)@B, (mx-mx_)@B)
        
        return p.ravel()
    
    @staticmethod
    def pearcorr2d(input1, input2):
        # input1: nchn, nspl
        # input2: nchn, nspl
        input1 = input1-np.mean(input1, axis=(0,1))
        input2 = input2-np.mean(input1, axis=(0,1))
        cc=np.sum(input1*input2,axis=(0,1))
        c1=np.sum(input1*input1,axis=(0,1))
        c2=np.sum(input2*input2,axis=(0,1))
        return cc/np.sqrt(c1*c2)
    
class FBTRCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, frequencies, optimizer='trca', ifG=True):
        super(FBTRCA, self).__init__()
        self.strcas = []
        self.K = None
        self.optimizer=optimizer
        self.num_out = num_out
        self.ifG = ifG
        self.frequencies = frequencies
        for _ in enumerate(self.frequencies):
            self.strcas.append(STRCA(Co=self.num_out, optimizer=self.optimizer, ifG=self.ifG))
            
    def fit(self, X:np.ndarray, y:np.ndarray):
        # X: trial, channel, time, frequenecy
        # y: trial
        assert len(self.frequencies)==X.shape[-1]
        y = y.ravel()
        # X = zscore(X, 2, 1)
        for i, _ in enumerate(self.frequencies):
            self.strcas[i].fit(X[:,:,:,i], y)
        self.K = self.strcas[0].K
        return self
            
    def transform(self, X:np.ndarray):
        pattern = np.zeros((X.shape[0], self.K*8, len(self.frequencies)))
        for i, _ in enumerate(self.frequencies):
            pattern[:,:,i] = self.strcas[i].transform(X[:,:,:,i])
        return pattern.reshape(X.shape[0], self.K*8*len(self.frequencies))
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class ZNormal(BaseEstimator, TransformerMixin):
    def __init__(self, dim):
        super(ZNormal, self).__init__()
        self.dim=dim
            
    def fit(self, X:np.ndarray, y:np.ndarray):
        return self
            
    def transform(self, X:np.ndarray):
        # n_trl, nchn, nspl
        u = np.mean(X, self.dim, keepdims=True)
        v = np.std(X, self.dim, ddof=1, keepdims=True)
        X = (X-u)/v
        u = np.reshape(u, (u.shape[0], -1))
        v = np.reshape(v, (v.shape[0], -1))
        return X, u, v
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class ZSTRCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, frequency, zdim, optimizer='trca', ifG=True, ifZ=True, btype='lowpass', fs=256, order=8):
        super(ZSTRCA, self).__init__()
        self.num_out = num_out
        self.frequency = frequency
        self.zdim = zdim
        self.optimizer = optimizer
        self.ifG = ifG

        self.znorm = ZNormal(self.zdim)
        self.strc = STRCA(self.num_out, self.optimizer, self.ifG)
        self.btype = btype
        self.ifZ = ifZ
        self.fs = fs
        self.order = order

    def fit(self, X:np.ndarray, y:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.fit_transform(X)
        X = self.filter_data(X)
        self.strc.fit(X, y)
        return self
            
    def transform(self, X:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.transform(X)
        X = self.filter_data(X)
        if self.ifZ:
            X = self.strc.transform(X).reshape((X.shape[0],-1))
            X = np.concatenate((X, u.reshape(u.shape[0],-1), v.reshape(v.shape[0], -1)), axis=1)
        else:
            X = self.strc.transform(X).reshape((X.shape[0],-1))
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def filter_data(self, X):
        # n_trl, nchn, nspl

        if self.btype=='lowpass':
            sos = butter(self.order, self.frequency, fs=self.fs, btype=self.btype, output='sos')
        elif self.btype=='bandpass':
            sos = butter(self.order, [0.05, self.frequency], fs=self.fs, btype=self.btype, output='sos')

        mirror = np.flip(X, axis=2)
        mirsigror = np.concatenate((mirror[:,:,-int(np.sqrt(self.fs)):], X,
                                mirror[:,:,:int(np.sqrt(self.fs))]), axis=2)
        Xr = sosfiltfilt(sos, mirsigror, axis=2)[:,:,int(np.sqrt(self.fs)):-int(np.sqrt(self.fs))]
        return Xr

class ZFBTRCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, frequencies, zdim, optimizer='trca', ifG=True, ifZ=True, btype='lowpass', fs=256, order=8):
        super(ZFBTRCA, self).__init__()
        self.num_out = num_out
        self.frequencies = frequencies
        self.zdim = zdim
        self.optimizer = optimizer
        self.ifG = ifG

        self.znorm = ZNormal(self.zdim)
        self.fbtrc = FBTRCA(self.num_out, self.frequencies, self.optimizer, self.ifG)
        self.btype = btype
        self.ifZ = ifZ
        self.fs = fs
        self.order = order

    def fit(self, X:np.ndarray, y:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.fit_transform(X)
        X = self.filter_data(X)
        self.fbtrc.fit(X, y)
        return self
            
    def transform(self, X:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.transform(X)
        X = self.filter_data(X)
        if self.ifZ:
            X = self.fbtrc.transform(X).reshape((X.shape[0],-1))
            X = np.concatenate((X, u.reshape(u.shape[0],-1), v.reshape(v.shape[0], -1)), axis=1)
        else:
            X = self.fbtrc.transform(X).reshape((X.shape[0],-1))
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def filter_data(self, X):
        # n_trl, nchn, nspl

        for f in np.arange(len(self.frequencies)):
            if self.btype=='lowpass':
                sos = butter(self.order, self.frequencies[f], fs=self.fs, btype=self.btype, output='sos')
            elif self.btype=='bandpass':
                sos = butter(self.order, [0.05, self.frequencies[f]], fs=self.fs, btype=self.btype, output='sos')

            mirror = np.flip(X, axis=2)
            mirsigror = np.concatenate((mirror[:,:,-int(np.sqrt(self.fs)):], X,
                                    mirror[:,:,:int(np.sqrt(self.fs))]), axis=2)
            tmp_fdata = np.expand_dims(
                sosfiltfilt(sos, mirsigror, axis=2)[:,:,int(np.sqrt(self.fs)):-int(np.sqrt(self.fs))]
                ,3)
            Xr = tmp_fdata if f==0 else np.concatenate([Xr, tmp_fdata], axis=3)
        return Xr

class EEGNet(torch.nn.Module):
    def __init__(self,
                 num_samples: int = 512,
                 num_channels: int = 11,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = torch.nn.Sequential(
            torch.nn.ZeroPad2d([self.kernel_1//2-1,self.kernel_1//2,0,0]),
            torch.nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, bias=False),
            torch.nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            torch.nn.Conv2d(self.F1,
                      self.F1 * self.D, (self.num_channels, 1),
                      groups=self.F1,
                      bias=False), 
            torch.nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            torch.nn.ELU(), 
            torch.nn.AvgPool2d((1, 4), stride=4), 
            torch.nn.Dropout(p=dropout),
            torch.nn.ZeroPad2d([self.kernel_2//2-1,self.kernel_2//2,0,0]),
            torch.nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D),
            torch.nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            torch.nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), 
            torch.nn.ELU(), 
            torch.nn.AvgPool2d((1, 8), stride=8),
            torch.nn.Dropout(p=dropout)
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(self.F1 * self.D*self.num_samples//32, num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x
    
class TEGNet(torch.nn.Module):
    def __init__(self,
                 optimizer: str='tdlda',
                 num_samples: int = 512,
                 num_channels: int = 11,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(TEGNet, self).__init__()
        self.chnopt = ChannelOptimizer(num_channels, optimizer)
        self.chnoptW = None
        self.depnet = EEGNet(num_samples, num_channels, F1, F2, D, num_classes, kernel_1, kernel_2, dropout)
    
    def initialize(self, X, y):
        # X: trial, channel, time
        self.chnopt.fit(X, y)
        self.chnoptW = torch.nn.Parameter(torch.Tensor(self.chnopt.W), requires_grad=False)
    
    def forward(self, x: torch.Tensor):
        x = torch.einsum('co, bfct->bfot', self.chnoptW, x)
        return self.depnet(x)
    
class Classifier(torch.nn.Module):
    def __init__(self, num_fbanks, num_classes):
        super(Classifier, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(num_fbanks*num_classes,num_fbanks*num_classes*2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(num_fbanks*num_classes*2,num_fbanks*num_classes//2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(num_fbanks*num_classes//2,num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x
    
class OTSNet(torch.nn.Module):
    def __init__(self,
                num_fbanks: int=10,
                num_samples: int = 768,
                num_channels: int = 11,
                F1: int = 8,
                F2: int = 16,
                D: int = 2,
                num_classes: int = 2,
                kernel_1: int = 64,
                kernel_2: int = 16,
                dropout: float = 0.25):
        super(OTSNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.filterbanks = [nfbank for nfbank in range(num_fbanks)]

        self.block1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                EEGNet(
                    num_samples=self.num_samples,
                    num_channels=self.num_channels,
                    F1=self.F1,
                    F2=self.F2,
                    D=self.D,
                    num_classes=self.num_classes,
                    kernel_1=self.kernel_1,
                    kernel_2=self.kernel_2,
                    dropout=self.dropout,
                )
            ) for _ in self.filterbanks
            ])
        self.block2 = Classifier(num_fbanks, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.block1[netindex](x[:,netindex,:,:].unsqueeze(1)).unsqueeze(2) for netindex in range(len(self.block1))], dim=2)
        x = self.block2(x)
        return x

class TTSNet(OTSNet):
    def __init__(self,
                num_fbanks: int=10,
                num_samples: int = 768,
                num_channels: int = 11,
                F1: int = 8,
                F2: int = 16,
                D: int = 2,
                num_classes: int = 2,
                kernel_1: int = 64,
                kernel_2: int = 16,
                dropout: float = 0.25):
        super(TTSNet, self).__init__(num_fbanks, num_samples, num_channels, F1,
                        F2, D, num_classes, kernel_1, kernel_2, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.block1[netindex](x[:,netindex,:,:].unsqueeze(1)).unsqueeze(2) for netindex in range(len(self.block1))], dim=2)
        if self.training:
            x = torch.cat([x, self.block2(x).unsqueeze(-1)], dim=-1)
        else:
            x = self.block2(x)
        return x

class OTSLoss(torch.nn.Module):
    def __init__(self):
        super(OTSLoss, self).__init__()
        self.celoss = torch.nn.CrossEntropyLoss()
        
    def forward(self, predicted, target):
        loss = self.celoss(predicted, target)
        return loss

class TTSLoss(OTSLoss):
    def __init__(self):
        super(TTSLoss, self).__init__()
        
    def forward(self, predicted, target):
        if self.training:
            target = torch.repeat_interleave(torch.reshape(target, [-1, 1]), repeats=predicted.shape[-1], dim=1).ravel()
            predicted = torch.transpose(predicted, 1, -1).reshape((-1, predicted.shape[1]))
        loss = self.celoss(predicted, target)
        return loss