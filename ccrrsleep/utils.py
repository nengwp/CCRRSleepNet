import itertools
import numpy as np
from collections import Counter
# from scipy.ndimage.interpolation import shift
from scipy.ndimage import shift



def get_balance_class_downsample(x, y):
    """
    Balance the number of samples of all classes by (downsampling):
        1. Find the class that has a smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.random.permutation(idx)[:n_min_classes]
        balance_x.append(x[idx])
        balance_y.append(y[idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """

    from imblearn.over_sampling import SMOTE, BorderlineSMOTE,RandomOverSampler
    from imblearn.under_sampling  import RandomUnderSampler
    # sm = SMOTE(random_state=1, n_jobs=32)
    sm = BorderlineSMOTE(random_state=1, kind="borderline-1", n_jobs=32)
    # sm = RandomOverSampler(random_state=1)
    # sm = RandomUnderSampler(random_state=1)

    x = np.squeeze(x)
    balance_x, balance_y = sm.fit_sample(x, y)
    balance_x = np.expand_dims(balance_x, axis=2)
    balance_x = np.expand_dims(balance_x, axis=3)
    return balance_x, balance_y

########
def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if shuffle:
            yield DataAugmentation(inputs[excerpt]), targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt]


def iterate_seq_minibatches(inputs, targets, batch_size, seq_length, stride):
    """
    Generate a generator that return a batch of sequence inputs and targets.
    """
    assert len(inputs) == len(targets)
    n_loads = (batch_size * stride) + (seq_length - stride)
    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:],
                              dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        for b_idx in range(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1,) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1,) + targets.shape[1:])
        yield flatten_inputs, flatten_targets

############
def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y


def iterate_list_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    for idx, each_data in enumerate(zip(inputs, targets)):
        each_x, each_y = each_data
        seq_x, seq_y = [], []
        for x_batch, y_batch in iterate_seq_minibatches(inputs=each_x, 
                                                        targets=each_y, 
                                                        batch_size=1, 
                                                        seq_length=seq_length, 
                                                        stride=1):
            seq_x.append(x_batch)
            seq_y.append(y_batch)
        seq_x = np.vstack(seq_x)
        seq_x = seq_x.reshape((-1, seq_length) + seq_x.shape[1:])
        seq_y = np.hstack(seq_y)
        seq_y = seq_y.reshape((-1, seq_length) + seq_y.shape[1:])
        
        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=seq_x, 
                                                              targets=seq_y, 
                                                              batch_size=batch_size, 
                                                              seq_length=1):
            x_batch = x_batch.reshape((-1,) + x_batch.shape[2:])
            y_batch = y_batch.reshape((-1,) + y_batch.shape[2:])
            yield x_batch, y_batch



def DataAugmentation(x, roll_range=0.5, horizontal_flip=True,seed=None):
    assert x.shape[1:] == (3000, 1, 1)
    if seed is not None:
        np.random.seed(seed)
    N = x.shape

    if roll_range:
        if np.random.random() < 0.5:
            tx = np.random.uniform(-roll_range, roll_range)
            if roll_range < 1:
                tx *= N[1]
            x = np.roll(x, int(tx), axis=1)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = np.flip(x,axis=1)
    return x


def sequence_down_sample(data,lable,sequence_length=25, down_rate=5):
    lab_dict = Counter(lable)
    lab_list = []
    for i in lab_dict:
        lab_i = np.array([i, lab_dict[i]])
        lab_list.append(lab_i)
    lab_list = np.array(lab_list)
    lab_array = lab_list[np.argsort(lab_list[:, 0])]

    data_list = []
    lab_list = []
    # 10 最小控制数 batch_size*down_rate

    min_num = int(lab_array[:, 1].min()/down_rate)
    min_num = min_num if min_num>1 else int(lab_array[:, 1].min())

    for i in range(len(lab_array)):
        lab = lab_array[i, 0]
        data_idx = np.array(np.where(lable == lab)).reshape(-1)
        np.random.shuffle(data_idx)
        data_idx = data_idx[0:int(min_num)]

        for j in range(int(min_num)):
            N = sequence_length
            bias = np.random.randint(int(-N / 2), int(N / 2))
            if data_idx[j] + bias + N > data.shape[0]:
                bias = -N
            if data_idx[j] + bias < 0:
                bias = N
            data_ij = data[data_idx[j] + bias:data_idx[j] + bias + N, :]
            lab_ij = lable[data_idx[j] + bias:data_idx[j] + bias + N]
            data_list.append(data_ij)
            lab_list.append(lab_ij)
    data_sample = np.concatenate(tuple([data_i for data_i in data_list]), axis=0)
    lab_sample = np.concatenate(tuple([lab_i for lab_i in lab_list]), axis=0)
    return data_sample,lab_sample