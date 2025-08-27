import numpy as np
from h5py import File
from scipy.stats import zscore
from scipy.signal import butter, sosfiltfilt, detrend
from ssqueezepy import cwt
class Dataset(object):
    def __init__(self, num_class, path, channel=None) -> None:
        self.data = []
        self.label = []
        self.path = path
        self.index_train = []
        self.index_test = []
        self.num_class = num_class
        self.channel = channel
        self.filter_bank_A = []
        self.filter_bank_B = []

    def load_data(self, sub, marker = None):
        for mn in np.arange(self.num_class):
            f = File(self.path+'sub'+str(sub)+'_mn'+str(mn)+'.hdf5','r')
            tmp_data = np.array(f['data'])
            self.channel = np.arange(tmp_data.shape[1]) if self.channel is None else self.channel
            if np.isnan(tmp_data).any():
                m_trl,m_chn,m_spl = np.where(np.isnan(tmp_data))
                missing_entries = np.unique([[d1,d2] for d1,d2 in zip(m_trl,m_chn)], axis=0)
                for i, j in missing_entries:
                    tmp_data[i,j,np.isnan(tmp_data[i,j])]=np.nanmean(tmp_data[i,:,np.isnan(tmp_data[i,j])], axis=1)
            self.data.append(self.scale_wrapper(
                    detrend(tmp_data[:,self.channel,:], axis=2, type='linear'), marker=marker))
            
            self.label.append(np.ones([np.shape(self.data[mn])[0]]) * mn)
            f.close()

    @staticmethod
    def scale_wrapper(X, marker = None):
        if marker == None:
            return X
        elif marker == 'zscore':
            return zscore(X, axis=-1)
    
    def divide_data(self, nfold):
        for mn in np.arange(self.num_class):
            train, test = self.random_slice(self.label[mn],nfold)
            self.index_train.append(train)
            self.index_test.append(test)

    def initial_filter_data(self):
        for mn in np.arange(self.num_class):
            self.data[mn] = self.data[mn][:,:,:,0]

    def filter_data(self, fs, frange, order=8, btype='lowpass', bplow=0.01):
        for mn in np.arange(self.num_class):
            if self.data[mn].ndim == 3:
                self.data[mn] = np.expand_dims(self.data[mn],3)
        for f in np.arange(len(frange)):
            if btype=='lowpass':
                sos = butter(order, frange[f], fs=fs, btype=btype, output='sos')
            elif btype=='bandpass':
                sos = butter(order, [bplow, frange[f]], fs=fs, btype=btype, output='sos')
            for mn in np.arange(self.num_class):
                mirror = np.flip(self.data[mn][:,:,:,0], axis=2)
                mirsigror = np.concatenate((mirror[:,:,-int(np.sqrt(fs)):], 
                                      self.data[mn][:,:,:,0],
                                      mirror[:,:,:int(np.sqrt(fs))]), axis=2)
                tmp_fdata = np.expand_dims(
                    sosfiltfilt(sos, mirsigror, axis=2)[:,:,int(np.sqrt(fs)):-int(np.sqrt(fs))]
                    ,3)
                self.data[mn] = np.concatenate([self.data[mn], tmp_fdata], axis=3)
    
    def retreive_data(self, class_index, fold_index, filter_index, precison=None):
        for ind in np.arange(len(class_index)):
            cls = class_index[ind]
            train_index = self.index_train[cls][fold_index,:]
            test_index = self.index_test[cls][fold_index,:]
            if ind == 0:
                if  self.data[cls].ndim == 3:
                    train_data = self.data[cls][train_index]
                    test_data = self.data[cls][test_index]
                else:
                    train_data = self.data[cls][train_index][:,:,:,filter_index]
                    test_data = self.data[cls][test_index][:,:,:,filter_index]
                train_label = self.label[cls][train_index]
                test_label = self.label[cls][test_index]
            else:
                if  self.data[cls].ndim == 3:
                    train_data = np.concatenate([train_data, self.data[cls][train_index]], axis=0)
                    test_data = np.concatenate([test_data, self.data[cls][test_index]], axis=0)
                else:
                    train_data = np.concatenate([train_data, self.data[cls][train_index][:,:,:,filter_index]], axis=0)
                    test_data = np.concatenate([test_data, self.data[cls][test_index][:,:,:,filter_index]], axis=0)
                train_label = np.concatenate([train_label, self.label[cls][train_index]], axis=0).ravel()
                test_label = np.concatenate([test_label, self.label[cls][test_index]], axis=0).ravel()

        if precison is not None:
            train_data = train_data.astype(np.float32)
            train_label= train_label.astype(np.int64)
            test_data  = test_data.astype(np.float32)
            test_label = test_label.astype(np.int64)
        return train_data, train_label, test_data, test_label
    
    @staticmethod
    def random_slice(label,n):
        [C, class_counts] = np.unique(label, return_counts=True)
        num_class  = len(C)
        num_test   = np.int64(np.floor(class_counts/n))
        num_train  = class_counts-num_test
        index_test =np.zeros([n, int(np.sum(num_test))])
        index_train=np.zeros([n, int(np.sum(num_train))])
        for c in np.arange(num_class):
            np.random.seed(c)
            tmp_perm=np.random.permutation(class_counts[c])
            for count_n in np.arange(n):
                tmp=tmp_perm[count_n*num_test[c]:(count_n+1)*num_test[c]]
                index_test[count_n, np.sum(num_test[:c]):np.sum(num_test[:c+1])]=np.sum(class_counts[:c])+tmp
                index_train[count_n, np.sum(num_train[:c]):np.sum(num_train[:c+1])]=np.sum(class_counts[:c])+np.setdiff1d(np.arange(class_counts[c]), tmp)
        return np.int64(index_train), np.int64(index_test)