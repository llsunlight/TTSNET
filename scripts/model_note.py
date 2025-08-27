import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import qr, svd, solve, eigh
from scipy.signal import butter, sosfiltfilt

'''
这段代码分为三个部分：
第一部分是确定滤波器的类型；
第二部分是用相应的滤波器处理数据transformer;
第三部分
'''

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
        self.optimizer = optimizer #通道变换工具

    def fit(self, X:np.ndarray, y:np.ndarray):
        # X: trial, channel, time
        # y: trial
        y = y.ravel()  #展平数组y
        _, self.Ci,  self.T = X.shape #获取数据维度
        category = sorted(np.unique(y))#unique函数提取不重复特征值并升序排序
        self.K = len(category)
        self.M = np.zeros((self.K, self.Ci, self.T))#创建指定大小的全零三位数组
        optimizer = self.optimizer
        if optimizer is None or optimizer.lower() == 'none':#不优化，用原通道作为输出
            self.W = np.eye(self.Ci)#[:, self.Ci//2-self.Co//2:self.Ci//2+1+self.Co//2] 生成单位矩阵
            for i, c in enumerate(category):
                x = X[y==c]
                self.M[i] = x.mean(0) #每一个特征值对应的通道数和时间点数求和
            self.G = self.M.mean(0)#所有特征值的“平均信号”
        elif optimizer == 'trca':#找一个空间滤波器，使不同trial在同一类别下时间一致性最强
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
        elif optimizer == 'tdlda':#LDA思想
            self.G = X.mean(0) #平均信号
            self.S = np.zeros((self.K, self.Ci, self.Ci))
            self.Q = np.zeros((self.K, self.Ci, self.Ci))
            for i, c in enumerate(category):
                x = X[y==c]
                self.M[i] = x.mean(0) #每个类型的平均信号
                self.S[i] = (self.M[i]-self.G)@(self.M[i]-self.G).T #构造类间协方差矩阵
                for xx in x:
                    self.Q[i] += (xx-self.M[i])@(xx-self.M[i]).T #构造类内协方差矩阵
            _, self.W = eigh(self.S.mean(0), self.Q.mean(0), driver='gv', check_finite=False) #找出最佳的判别方向
            #self.W是一个二维矩阵，每一列对应一个方向向量，也就是一个空间滤波器

            # W is sorted in the ascending order now, we need to reverse it
            self.W = self.W[:,::-1] 
            # self.W = self.geigensolve(self.S.mean(0), self.Q.mean(0))
            self.W = self.W[:self.Ci, :self.Co] #保留最重要的Co个方向
        return self

    def transform(self, X:np.ndarray):  #用已经学习好的空间滤波器转化新的EEG
        # X: trial, channel, time
        W = self.W
        if len(X.shape)==2:  #保证传入的数据都是三维的
            X = np.expand_dims(X, 0)
        newX = np.einsum('ij,kjl->kil',W.T,X) #用W的转置矩阵，做ij,kjl->kil变换，使数据trail变为W中的trail
        return np.expand_dims(newX, 1) #保证输出的数据都是三维的
    
    def fit_transform(self, X, y=None): #将fit和transform合并到一起
        self.fit(X, y)
        return self.transform(X)
    
    #一种静态方法，通过直接类调用，而不需要实例化对象
    @staticmethod
    def trca(data:np.ndarray):
        # trial, channel, time
        n_l, n_c, n_t  = data.shape
        UX = np.swapaxes(data, 0, 1).reshape((n_c, n_l*n_t)) #将channel和time合并成一维
        UX = UX - np.mean(UX, axis=1, keepdims=True) #每一行减去time维度的平均值，去中心化处理
        Q = UX@UX.T/(n_t*n_l)

        data -= np.mean(data, axis=2, keepdims=True) #对每个 trial 的每个通道，在时间维上减去自身均值，结果保持 shape 相同
        U = data.sum(0) #在trail求和
        V = np.zeros((n_c, n_c)) 
        for k in range(n_l):
            V += data[k]@data[k].T #取每个trail中channel time平均矩阵协方差的和
        S = (U@U.T-V)/(n_t*n_l*(n_l-1)) #归一化处理
        return S, Q
    
    #求解广义特征值
    @staticmethod
    def geigensolve(S, Q):
        [d,v] =np.linalg.eig(Q) #求Q的特征向量和特征值
        d=np.real(d) #把所有特征值变为实数
        index=np.argsort(-d) #将特征值从大到小排序，并将对应的特征向量排序
        d=d[index]
        v=v[:,index]
        P=np.diag(1.0/np.sqrt(d))*v.T #将Q转化为单位矩阵
        T=P@S@P.T #在“白化空间”中变成一个新的矩阵
        [d,v] =np.linalg.eig(T) #对T再做一次特征分解
        d=np.real(d)
        index=np.argsort(-np.abs(d))
        v=v[:,index]
        w=P.T@v #将找出对最优结果投影回原始空间中
        W=w/np.linalg.norm(w,2,axis=0,keepdims=True) #对每列向量进行归一化，使其长度为1
        return W
    

class STRCA(ChannelOptimizer, BaseEstimator, TransformerMixin):
    def __init__(self, Co, optimizer, ifG):
        super(STRCA, self).__init__(Co, optimizer)
        self.optimizer = optimizer
        self.ifG = ifG

    def fit(self, X:np.ndarray, y:np.ndarray):
        super(STRCA, self).fit(X, y) #直接调用父类中的fit完成通道优化

    def transform(self, X:np.ndarray, ifG=True):
        # X: trial, channel, time
        W = self.W
        if self.ifG:
            G = self.G 
        else:
            G = np.zeros_like(self.G) #平均总值
        if len(X.shape)==2:
            X = np.expand_dims(X, 0)
        pattern = np.zeros((X.shape[0], self.K, 8)) #8维特征，2行4列拉平
        for i in range(X.shape[0]):
            for c in range(self.K):
                x  = X[i,:,:]-G
                mx = self.M[c,:,:]-G #目标类的平均模板
                mx_= (self.M.sum(0)-self.M[c,:,:])/(self.K-1)-G #非目标类的平均模板
                pattern[i,c,:]=self.corrfea(x.T, mx.T, mx_.T, W)
        return pattern.reshape((X.shape[0], -1))
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    @staticmethod
    def canoncorr(x:np.ndarray, y:np.ndarray):
        n, p1 = x.shape #当前样本与目标类均值的皮尔逊相关性
        _, p2 = y.shape #加权通道后的相关性（使用 W）d

        [Q1, R1, perm1] = qr(x, mode='economic', pivoting=True) #带列交换的QR分解（pivoting=True）
        rankX = np.sum(np.abs(np.diag(R1))>np.finfo(R1.dtype).eps*max(n,p1)) #估计矩阵x的秩（rank）
        if rankX < p1: #如果 x 不是满秩（即有冗余特征），裁剪掉无效列，只保留有效秩部分
            Q1 = Q1[:,:rankX]
            R1 = R1[:rankX, :rankX]
        
        # 接下来，对 y 做同样的处理
        [Q2, R2, perm2] = qr(y, mode='economic', pivoting=True)
        rankY = np.sum(np.abs(np.diag(R2))>np.finfo(R2.dtype).eps*n*p2)
        if rankY < p2:
            Q2 = Q2[:,:rankY]
            R2 = R2[:rankY, :rankY]
        # 取 x 和 y 中较小的秩   
        d = min(rankX, rankY)
        #Q1TQ2 的SVD本质上找出了x和y在正交子空间上最相关的方向
        [U, S, Vh] = svd(Q1.T@Q2, full_matrices=False, compute_uv=True, lapack_driver='gesvd') 
        #通过解线性方程得到原空间下的变换矩阵 A 和 B 
        A = solve(R1, U[:,:d]*np.sqrt(n-1))
        B = solve(R2, Vh[:d,:].T*np.sqrt(n-1))
        #还原QR分解时 重新排列的列 补0
        A[perm1, :] = np.concatenate((A, np.zeros((p1-rankX, d))), axis=0)
        B[perm2, :] = np.concatenate((B, np.zeros((p2-rankY, d))), axis=0)

        return A, B
    
    #特征提取 EEG 信号（或其他脑电信号）提炼成更小、更有区分力的特征向量
    def corrfea(self, x:np.ndarray, mx:np.ndarray, mx_:np.ndarray, W:np.ndarray):
        # x  : time, channel
        # mx : time, channel, target class
        # mx_: time, channel, mean of other classes
        p = np.zeros((2,4))
        p[0, 0] = self.pearcorr2d(x, mx) #没加权前，直接比较原信号 x 和目标均值 mx 的皮尔逊相关系数
        p[0, 1] = self.pearcorr2d(x@W, mx@W) ### 加权，信号增强、降噪后比较 1
        # A, B    = self.canoncorr(x, mx)
        # p[0, 2] = self.pearcorr2d(x@A, mx@A)
        # p[0, 3] = self.pearcorr2d(x@B, mx@B) ### 3
        
        p[1, 0] = self.pearcorr2d(x-mx_, mx-mx_) #把 x 和 mx 都减掉其他类别均值 mx_，再比相关性，消除其他类的公共部分
        p[1, 1] = self.pearcorr2d((x-mx_)@W, (mx-mx_)@W)
        # A, B    = self.canoncorr(x-mx_, mx-mx_)
        # p[1, 2] = self.pearcorr2d((x-mx_)@A, (mx-mx_)@A) ### 6
        # p[1, 3] = self.pearcorr2d((x-mx_)@B, (mx-mx_)@B)
        
        return p.ravel() #把 p 拉平成一维向量
    
    @staticmethod
    def pearcorr2d(input1, input2):
        # input1: nchn, nspl 通道数 采样点数
        # input2: nchn, nspl
        input1 = input1-np.mean(input1, axis=(0,1)) 
        input2 = input2-np.mean(input1, axis=(0,1))
        cc=np.sum(input1*input2,axis=(0,1)) #协方差分子
        c1=np.sum(input1*input1,axis=(0,1)) #标准差分母
        c2=np.sum(input2*input2,axis=(0,1))
        return cc/np.sqrt(c1*c2) #返回皮尔逊相关系数 = 协方差/标准差乘积
    
class FBTRCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, frequencies, optimizer='trca', ifG=True):
        super(FBTRCA, self).__init__()
        self.strcas = []
        self.K = None
        self.optimizer=optimizer
        self.num_out = num_out
        self.ifG = ifG
        self.frequencies = frequencies
        for _ in enumerate(self.frequencies): #为每一个频带新建一个STRCA实例
            self.strcas.append(STRCA(Co=self.num_out, optimizer=self.optimizer, ifG=self.ifG))

    #在每个频带（每个滤波器输出）上，分别训练各自的 STRCA 模型      
    def fit(self, X:np.ndarray, y:np.ndarray):
        # X: trial, channel, time, frequenecy
        # y: trial
        assert len(self.frequencies)==X.shape[-1] #频带数量 frequencies = X的最后一维（频率数
        y = y.ravel()
        # X = zscore(X, 2, 1)
        for i, _ in enumerate(self.frequencies): #每个频带都独立训练一个STRCA模型
            self.strcas[i].fit(X[:,:,:,i], y)
        self.K = self.strcas[0].K #保存类别数 K
        return self
            
    def transform(self, X:np.ndarray): #trail 通道数 时间点数 频带数
        pattern = np.zeros((X.shape[0], self.K*8, len(self.frequencies))) # 试次数 每个类别提取 8 个特征 频带数 len(self.frequencies)
        for i, _ in enumerate(self.frequencies):
            pattern[:,:,i] = self.strcas[i].transform(X[:,:,:,i]) #用对应的STRCA提取i个频带的特征
        return pattern.reshape(X.shape[0], self.K*8*len(self.frequencies)) #试次数 总特征数
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class ZNormal(BaseEstimator, TransformerMixin):
    def __init__(self, dim): #dim决定在哪个轴上做标准化
        super(ZNormal, self).__init__()
        self.dim=dim
            
    def fit(self, X:np.ndarray, y:np.ndarray):
        return self
            
    def transform(self, X:np.ndarray):
        # n_trl, nchn, nspl
        u = np.mean(X, self.dim, keepdims=True) #沿着dim计算均值
        v = np.std(X, self.dim, ddof=1, keepdims=True) 
        X = (X-u)/v #正式做标准化 
        u = np.reshape(u, (u.shape[0], -1)) #(试次数, 特征数) 
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

        self.znorm = ZNormal(self.zdim) #新建一个ZNormal实例，专门负责标准化
        self.strc = STRCA(self.num_out, self.optimizer, self.ifG) #新建一个 STRCA 实例，负责特征提取
        self.btype = btype
        self.ifZ = ifZ
        self.fs = fs
        self.order = order

    def fit(self, X:np.ndarray, y:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.fit_transform(X)
        X = self.filter_data(X) #对数据进行滤波
        self.strc.fit(X, y) #标准化➕滤波后 送进STRCA模型
        return self
            
    def transform(self, X:np.ndarray):
        # n_trl, nchn, nspl
        X, u, v =self.znorm.transform(X) #使用znorm转换器对输入数据进行标准化，返回标准化后的 X 和参数 u, v
        X = self.filter_data(X) #滤波处理
        if self.ifZ:
            X = self.strc.transform(X).reshape((X.shape[0],-1)) #例如：从 (n_trl, nchn, nspl) 变为 (n_trl, nchn*nspl)
            X = np.concatenate((X, u.reshape(u.shape[0],-1), v.reshape(v.shape[0], -1)), axis=1)
        else:
            X = self.strc.transform(X).reshape((X.shape[0],-1))
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def filter_data(self, X):
        # n_trl, nchn, nspl
        # 设计滤波器
        if self.btype=='lowpass':
            sos = butter(self.order, self.frequency, fs=self.fs, btype=self.btype, output='sos')
        elif self.btype=='bandpass':
            sos = butter(self.order, [0.05, self.frequency], fs=self.fs, btype=self.btype, output='sos')
        # 镜像延拓处理
        mirror = np.flip(X, axis=2)
        mirsigror = np.concatenate((mirror[:,:,-int(np.sqrt(self.fs)):], X,
                                mirror[:,:,:int(np.sqrt(self.fs))]), axis=2)
        Xr = sosfiltfilt(sos, mirsigror, axis=2)[:,:,int(np.sqrt(self.fs)):-int(np.sqrt(self.fs))]
        return Xr

class ZFBTRCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_out, frequencies, zdim, optimizer='trca', ifG=True, ifZ=True, btype='lowpass', fs=256, order=8):
        super(ZFBTRCA, self).__init__() #继承基类初始化方法
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
    #fbtrc
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
    

"""
Input:      [B, 1, C, T]    # 通道C=11，时间点T=512

Block 1:
  Temporal Conv:  (1, 64)         → [B, F1, C, T]
  Spatial Conv:   (C, 1) Depthwise → [B, F1×D, 1, T]
  AvgPool(1, 4)                  → [B, F1×D, 1, T//4]

Block 2:
  Depthwise Conv: (1, 16)        → [B, F1×D, 1, T//4]
  Pointwise Conv: (1,1) → F2     → [B, F2, 1, T//4]
  AvgPool(1, 8)                  → [B, F2, 1, T//32]

Flatten:                        → [B, F2×T//32]
Linear:                         → [B, num_classes]
"""

class EEGNet(torch.nn.Module):
    def __init__(self,
                 num_samples: int = 512,  #输入信号的时间点数（采样点数）
                 num_channels: int = 11,  #EEG通道数
                 F1: int = 8,             #第一层卷积核数量（空间滤波器数量）
                 F2: int = 16,            #第二层卷积核数量（时间滤波器数量）
                 D: int = 2,              #深度可分离卷积的深度因子（Depth multiplie)
                 num_classes: int = 2,    #输出类别数
                 kernel_1: int = 64,      #第一层卷积核大小
                 kernel_2: int = 16,      #第二层卷积核大小
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples #输入信号采样数（在时间轴上的长度）
        self.num_classes = num_classes #分类类别数
        self.num_channels = num_channels #输入通道数
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout


        self.block1 = torch.nn.Sequential(

            torch.nn.ZeroPad2d([self.kernel_1//2 - 1, self.kernel_1//2, 0, 0]),  # 如果补零较少会导致，补0位数不太够 将补零放到池化之后
            torch.nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, bias=False), #对时间轴做卷积不改变通道数
            torch.nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3), #归一化
            torch.nn.Conv2d(self.F1,
                      self.F1 * self.D, (self.num_channels, 1),
                      groups=self.F1,
                      bias=False),  #空间卷积
            torch.nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3), #归一化
            torch.nn.AvgPool2d(kernel_size=(1, 32), stride=32),#
            torch.nn.ELU(), 
            torch.nn.AvgPool2d((1, 4), stride=4), 
            torch.nn.Dropout(p=dropout),

            #第二阶段处理：深度可分离卷积
            torch.nn.ZeroPad2d([self.kernel_2//2-1,self.kernel_2//2,0,0]),
            torch.nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D),     #组卷积 对什么分组？
            torch.nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1), #pointwise卷积（1x1卷积）
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
                ),
            ) for _ in self.filterbanks
            ])
        self.block2 = Classifier(num_fbanks, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.block1[i](x[:,i,:,:].unsqueeze(1)) for i in range(len(self.block1))], dim=2)#增加池化层以后不需要通过标签提取
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
        x = torch.cat([self.block1[i](x[:,i,:,:].unsqueeze(1)) for i in range(len(self.block1))], dim=2)#增加池化层以后不需要通过标签提取
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