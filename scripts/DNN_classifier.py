import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import collections
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type = float, help = "specify number of epochs", default = 0.0001)
parser.add_argument('-ep', '--epochs', type = int, help = "specify number of epochs", default = 50)
parser.add_argument('-l', '--RNNlayers', type = int, help = "specify number of RNN layers", default = 3)
parser.add_argument('-nh', '--num_hidden_units', type = int, help = "specify number of hidden layer size", default = 128)
parser.add_argument('-fb', '--forget_bias', type = float, help = "specify forget bias for LSTM cell", default = 1)
parser.add_argument('-mx', '--Max_input_length', type = int, help = "The length of segment of each seq used", default = 10000)
parser.add_argument('-cn', '--num_conv_layer', type = str, help = "Number of conv layers along with number of filters per layer", default = '[48,64,96]')
parser.add_argument('-cl', '--length_conv_filter', type = str, help = "The length of conv layer filters", default = '[5,5,3]')
parser.add_argument('-gcn', '--gradient_clipping_norm', type = float, help = "Gradient clipping norm used during training.", default = 9.0)
parser.add_argument('-c', '--Class', type = str, help = "Specify protein dataset file name", default = 'cath-1.10.txt')
parser.add_argument('-dpr', '--drop_out_rate', type = float, help = 'set dropout rate', default = 0.3)
parser.add_argument('-up', '--upsampling', type = int, help = 'generate artificial samples, set 1 to activate this function', default = 0)
parser.add_argument('-sq', '--sqrt_trans', type = int, help = 'conduct square root transformation, set 1 to activate this function', default = 0)
parser.add_argument('-rr', '--remove_row_redundancy', type = int, help = 'only keep edges on the protein backbone', default = 0)
parser.add_argument('-a', '--alpha', type = float, help = 'parameter for artificial sample generating', default = 0.5)

parser.add_argument('-gpu', '--gpu_acceleration', type = int, help = 'Use GPU?', default = 1)
parser.add_argument('-br', '--Bidirectional_RNN', type = int, help = 'Use Bidirectional_RNN?', default = 1)

parser.add_argument('-tl','--title',type = str, help = "Specify Model Version or Name", default = 'CRNN_NEcv')
parser.add_argument('-k','--k_fold',type = int)
args = parser.parse_args()

#general setting
netcf = 6
upsampling = args.upsampling
squareroot_trans = args.sqrt_trans
amplification_idx = 1
rr = args.remove_row_redundancy
alpha = args.alpha

#parameter space
learning_rate = args.learning_rate
GradientClippingNorm = args.gradient_clipping_norm
epochs = args.epochs
RNNlayers = args.RNNlayers

#number of units in RNN cell
RNN_hidden_units = args.num_hidden_units
#decoder_hidden_units = 128

titleinfo = args.title

#forget_bias
fb = args.forget_bias

#max length
mxlen = args.Max_input_length

#CONV layer
def conv_param_ext(param):
    paramout = param.split(',')
    paramout = [x.replace('[','') for x in paramout]
    paramout = [int(x.replace(']','')) for x in paramout]
    return(paramout)

convlayer = conv_param_ext(args.num_conv_layer)
convsize = conv_param_ext(args.length_conv_filter)

#Choose class and datasets
Cout = args.Class.split('-')
#Cout = 'cath-1.10.txt'.split('-')
CL = Cout[0].upper()

#Introduce DropOut for CNN?
dp_rate = args.drop_out_rate

uGPU = True if args.gpu_acceleration==1 else False
biRNN = True if args.Bidirectional_RNN==1 else False
k = args.k_fold

time_start1 = time.time()
#Read the full available proteins list
allproteins = np.genfromtxt('./ProteinList/'+CL.lower()+'_6_list.txt', dtype=str)
#Data reading and processing

#read and process class table for proteins. Subset class table.
subclasstb = np.genfromtxt('./identifiers/'+ args.Class + '.txt', dtype = 'str')
Fullsamplesize = subclasstb.shape[0]
#check data availability
avail_picker = np.isin([x.split('.')[0] for x in subclasstb[:,1]], allproteins[:,0])
subclasstb = subclasstb[avail_picker]

#extract protein info from all proteins
info_picker = np.isin(allproteins[:,0], [x.split('.')[0] for x in subclasstb[:,1]])
subproteins = allproteins[info_picker]

subinfo = np.concatenate((subproteins[np.argsort(subproteins[:,0])], subclasstb[np.argsort(subclasstb[:,1])]), axis = 1)

proteinclass = [x for x in subinfo[:,2]]

pflen = subinfo.shape[0]

#this ypre needed to be one-hot transformed.
ypre = proteinclass
yct = dict(collections.Counter(ypre))

#remove class with only one obs
yk = list(yct.keys())
for itm in yk:
    if yct[itm] == 1:
        del yct[itm]

#further processing ypre remove entries occured only once.
subpicker = np.array([x for x in range(pflen) if ypre[x] in yct.keys()])

#all proteins are here now in subinfo
subinfonew = subinfo[subpicker]

ypre = [x for x in subinfonew[:,2]]
pflen = len(ypre)
Existsamplesize = pflen

def build_dict(inputs):
    dictionary = dict()
    for word in inputs:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

y_dict, rev_y_dict = build_dict(yct.keys())

#transfer y to one_hot coding
def input_one_hot(seq, vocab_size, dict):
    def one_hot_vec(num):
        y = np.zeros(vocab_size)
        y[num] = 1
        return y
    out = [one_hot_vec(dict[it]) for it in seq]
    return out

#insert cv & additional functions here, generate list first, then read or generate sample according to this list.

def system_sampling(seq, size, k, sid):
    #if cv size = 10, then k is from 0 to 9
    sl = len(seq)
    np.random.seed(sid)
    sysidx_pre = np.random.choice(min(sl,size), min(sl,size), replace = False)
    sysidx_pre2 = [x for x in sysidx_pre]
    quotient = math.floor(sl/min(sl,size))
    remaindr = sl%min(sl,size)
    sysidx = np.array(sysidx_pre2*quotient+sysidx_pre2[0:remaindr])
    trainpicker = sysidx != k%sl
    testpicker = sysidx == k%sl
    return trainpicker, testpicker

#further data processing: 5-fold-cv
def data_cv_pro(k, dt):
    idx_ori = np.arange(pflen)
    train_true_or_artificial = np.empty(0, dtype = int)

    trainlist = np.empty(0, dtype = int)
    testlist = np.empty(0, dtype = int)

    dict_sorted_by_value = sorted(dt.items(), key=lambda kv: kv[1], reverse = True)
    sorted_keys = [x[0] for x in dict_sorted_by_value]

    idx_1 = 0
    for itm in sorted_keys:
        idx_1 += 1
        yselect = idx_ori[np.array(ypre) == itm]
        trainpicker, testpicker = system_sampling(yselect, 5, k, 2016)
        idxtest = yselect[testpicker]
        idxtraining = yselect[trainpicker]
        idxtraining_toraidx = np.repeat(1, idxtraining.shape[0])

        trainlist = np.concatenate((trainlist,idxtraining))

        train_true_or_artificial = np.concatenate((train_true_or_artificial, idxtraining_toraidx))
        testlist = np.concatenate((testlist, idxtest))

        if idx_1 == 1:
            upb_sample_size = math.floor(idxtraining.shape[0]*amplification_idx)
            
        local_training_sample_size = idxtraining.shape[0]
        if upsampling == 1:
            if upb_sample_size-local_training_sample_size > 0:
                np.random.seed(2049)
                idxtraining_artificial = np.random.choice(idxtraining, size = upb_sample_size-local_training_sample_size)
                trainlist = np.concatenate((trainlist, idxtraining_artificial))
                train_true_or_artificial = np.concatenate((train_true_or_artificial, np.repeat(0, idxtraining_artificial.shape[0])))

    return trainlist, train_true_or_artificial, testlist

trainlist, train_true_or_artificial, testlist = data_cv_pro(k, y_dict)

def reorder(dat):
    ids = dat[:,[0,1]]

    for i in range(ids.shape[0]):
        if ids[i,0] > ids[i,1]:
            ids[i,[0,1]] = ids[i,[1,0]]

    orderidx = np.lexsort((ids[:,1], ids[:,0]))
    ids = ids[orderidx]
    dat = dat[orderidx]

    outidx = [x for x in range(ids.shape[0]) if ids[x,1]-ids[x,0]==1]
    out = dat[outidx]
    return out

def sample_reader(proteinid, arti_indi = 1, alpha = .5):
    basepath = './Results/BuildNet/'+args.Class+'/wt-GDVs-4-A/'

    if arti_indi == 1:
        xout = np.genfromtxt(basepath + subinfonew[proteinid,0] + '/CVM1', dtype = 'float')
        xlen = xout.shape[0]

        orderidx = np.lexsort((xout[:,1], xout[:,0]))
        xout = xout[orderidx]

        if rr == 1:
            xout = reorder(xout)

        if xlen > 4000 and uGPU == True:
            xout = xout[0:4000,:]
            xlen = xout.shape[0]

        xout = xout[:,2:75]
        
        if squareroot_trans == 1:
            xout = np.sqrt(xout)

        y = subinfonew[proteinid,2]

    elif arti_indi == 0:
        #this part has
        rout = np.genfromtxt(basepath + subinfonew[proteinid,0] + '/CVM1', dtype = 'float')
        sds = np.genfromtxt(glob.glob(basepath + subinfonew[proteinid,0] + '/CVM1_Count_*')[0], dtype = 'float')

        xout = np.empty(rout.shape)

        for a in range(rout.shape[0]):
            for b in range(rout.shape[1]):
                if b < 2:
                    xout[a,b] = rout[a,b]
                else:
                    xout[a,b] = np.random.normal(loc = rout[a,b], scale = alpha*np.sqrt(sds[a,b]), size = 1)
        
        xlen = xout.shape[0]

        orderidx = np.lexsort((xout[:,1], xout[:,0]))
        xout = xout[orderidx]

        if rr == 1:
            xout = reorder(xout)

        if rout.shape[0] > 4000 and uGPU == True:
            rout = rout[0:4000,:]
            sds = sds[0:4000,:]
            xlen = xout.shape[0]

        xout = xout[:,2:75]
        
        if squareroot_trans == 1:
            xout = np.sqrt(xout)
        y = subinfonew[proteinid,2]

    return xout, xlen, y

def data_generator(trainlist, train_true_or_artificial, testlist):
    #generate training samples
    xtraining = []
    xtraining_length = []
    ytraining = []

    for i in range(trainlist.shape[0]):
        xlocal, xlen, ylocal = sample_reader(trainlist[i], train_true_or_artificial[i], alpha = alpha)
        xtraining += [xlocal]
        xtraining_length += [xlen]
        ytraining += [ylocal]

    #generate test samples
    xtest = []
    xtest_length = []
    ytest = []

    for j in range(testlist.shape[0]):
        xlocal, xlen, ylocal = sample_reader(testlist[j])
        xtest += [xlocal]
        xtest_length += [xlen]
        ytest += [ylocal]

    return xtraining, xtraining_length, ytraining, xtest, xtest_length, ytest

x_train, x_train_length, y_train, x_test, x_test_length, y_test = data_generator(trainlist, train_true_or_artificial, testlist)

y_train = input_one_hot(y_train, len(y_dict), y_dict)
y_test = input_one_hot(y_test, len(y_dict), y_dict)

time_start2 = time.time()

print('epoch {}'.format(epochs))
print('RNNlayers {}'.format(RNNlayers))
print('RNN_hidden_units {}'.format(RNN_hidden_units))
print(titleinfo)
print('forget bias {}'.format(fb))

print('CONVlayer {}'.format(convlayer))
print('CONVSize {}'.format(convsize))
print('Class {}'.format(CL))

print('drop_out rate {}'.format(dp_rate))

#add additional artificial samples here.

#padding with zeros
def pad_zero(ele, max_length):
    cur_length = ele.shape[0]
    padzero = np.zeros(ele.shape[1]*(max_length-cur_length)).reshape([max_length-cur_length,ele.shape[1]])
    Out = np.concatenate((ele,padzero))
    return Out

x_train = [pad_zero(x, max(x_train_length+x_test_length)) for x in x_train]
x_test = [pad_zero(x, max(x_train_length+x_test_length)) for x in x_test]

#building model
ModelMode = True

#CNN_RNN start
#Place holder for Mini batch input
x = tf.placeholder("float", [None, max(x_train_length+x_test_length), 68])
y = tf.placeholder("float", [None, len(y_dict)])
length = tf.placeholder("int32", [None])

#CONV layers
def add_conv(x_in):
    conv = x_in 
    for i in range(len(convlayer)):
        conv_in = conv
        if i>0 and dp_rate > 0.0:
            conv_in = tf.layers.dropout(conv_in, rate = dp_rate, training = ModelMode)
        conv = tf.layers.conv1d(conv_in, filters = convlayer[i], kernel_size = convsize[i], activation = None, strides = 1, padding = 'same', name = "conv1d_{}".format(i))
    return conv

#LSTM(cudnn) layers
if uGPU:
    if biRNN:
        RNNdirect = 'bidirectional'
    else:
        RNNdirect = 'unidirectional'
    def RNN(conv):
        conv = tf.transpose(conv,[1,0,2])
        rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = RNNlayers, num_units = RNN_hidden_units, direction = RNNdirect, dropout = dp_rate if dp_rate>0.0 and ModelMode else 0.0)
        Outputs, _ = rnn_cell(conv)
        Outputs = tf.transpose(Outputs, [1,0,2])
        return Outputs
else:
    if biRNN:
        def RNN(conv):
            cell = tf.nn.rnn_cell.BasicLSTMCell
            cells_fw = [cell(RNN_hidden_units) for _ in range(RNNlayers)]
            cells_bw = [cell(RNN_hidden_units) for _ in range(RNNlayers)]
            if dp_rate>0.0 and ModelMode:
                cells_fw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1-dp_rate) for cell in cells_fw]
                cells_bw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1-dp_rate) for cell in cells_bw]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw = cells_bw, inputs=conv, sequence_length = length, dtype = tf.float32, scope = "rnn_classificaiton")
            return outputs
    else:
        def RNN(conv):
            def make_cell(lstm_size, fb):
                Out = rnn.BasicLSTMCell(lstm_size, forget_bias=fb)
                if dp_rate>0.0 and ModelMode:
                    Out_old = Out
                    Out = tf.contrib.rnn.DropoutWrapper(Out_old, output_keep_prob = 1-dp_rate)
                return Out
            rnn_cell = rnn.MultiRNNCell([make_cell(RNN_hidden_units, fb) for _ in range(RNNlayers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
            return outputs

def output_reducesum(output):
    mask = tf.tile(tf.expand_dims(tf.sequence_mask(length, tf.shape(output)[1]), 2), [1,1,tf.shape(output)[2]])
    zero_output = tf.where(mask, output, tf.zeros_like(output))
    output = tf.reduce_sum(zero_output, axis = 1)
    return output

def full_layer(output):
    #notice here that if 'bidirectional' is selected in RNN section, the weights here needs RNN layers*2
    if biRNN:
        weights = tf.Variable(tf.random_normal([RNN_hidden_units*2, len(y_dict)]))
    else:
        weights = tf.Variable(tf.random_normal([RNN_hidden_units, len(y_dict)]))
    bias = tf.Variable(tf.random_normal([len(y_dict)]))
    logit = tf.add(tf.matmul(output, weights),bias)
    return logit

#conduct the model
conv_out = add_conv(x)
rnn_out = RNN(conv_out)
mask_out = output_reducesum(rnn_out)

logits = full_layer(mask_out)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits))
optimizer = tf.contrib.layers.optimize_loss(loss = cross_entropy, global_step = tf.train.get_global_step(), learning_rate = learning_rate, optimizer='Adam', clip_gradients = GradientClippingNorm,
     summaries = ["learning_rate", "loss", "gradients", "gradient_norm"])

pred = tf.argmax(logits, axis = 1)
correct_pred = tf.equal(pred, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    accu_cul = []
    accu_by_type_cul = []
    test_accu_cul = []
    test_accu_by_type_cul = []
    x_test_len_cul = []

    def dict_comb(dict1, dict2):
        if not bool(dict1):
            dout = dict2
        else:
            dl = [dict1, dict2]
            dout = {}
            for k in dict1.keys():
                dout[k] = sum(d[k] for d in dl)
        return dout

    def accu_by_type(truth, pred):
        pred_cor_out = {}
        pred_count = {}
        pred_cpct_out ={}
        for ykey in y_dict.keys():
            pred_cor_out[ykey] = 0
            pred_count[ykey] = 0
            pred_cpct_out[ykey] = 0
        nl = len(pred)
        for i in range(nl):
            y_name = np.argmax(truth[i])
            pred_cor_out[rev_y_dict[y_name]] += int(pred[i]==y_name)
            pred_count[rev_y_dict[y_name]] += 1
        for ykey in y_dict.keys():
            pred_cpct_out[ykey] = pred_cor_out[ykey]/pred_count[ykey]
        return pred_cor_out, pred_count, pred_cpct_out

    def batch_distributor(cid, indx, x_in, y_in, len_in):
        subid = [x for x in range(len(indx)) if indx[x] == cid]
        xout = np.array([x_in[x] for x in subid])
        yout = np.array([y_in[x] for x in subid])
        lengthout = np.array([len_in[x] for x in subid])
        return {x:xout, y:yout, length:lengthout}

    for ep in range(epochs):
        #switch between training phase and test phase.
        ModelMode = True

        accu_cul_local = []
        pred_cul_local = []
        y_cul_local = []

        batch_size = min(32, len(x_test))
        len_x_train = len(x_train)
        n_batch = int(len_x_train/batch_size)
        indx = np.random.choice(n_batch, len_x_train, replace=True)
        indx_count = dict(collections.Counter(indx))
        for ib in range(n_batch):
            fd = batch_distributor(ib, indx, x_train, y_train, x_train_length)
            _ = session.run(optimizer, feed_dict=fd)
        for ib in range(n_batch):
            fd = batch_distributor(ib, indx, x_train, y_train, x_train_length)
            pd, accu = session.run([pred, accuracy], feed_dict=fd)
            #generate diagnostic information
            accu_cul_local += [accu*indx_count[ib]]
            ytrue = fd[y]
            if ib == 0:
                pred_cul_local = pd
                y_cul_local = fd[y]
            else:
                pred_cul_local = np.concatenate((pred_cul_local, pd))
                y_cul_local = np.concatenate((y_cul_local, fd[y]))

        accu_cul += [sum(accu_cul_local)/len_x_train]
        _, _, abt = accu_by_type(y_cul_local, pred_cul_local)
        accu_by_type_cul += [abt]

        if ep % 50 == 0:
            print('CV {}'.format(k))
            print('epoch {}'.format(ep))
            print(' training accuracy{}'.format(sum(accu_cul_local)/len_x_train))
            print(' Training accuracy by type{}'.format(abt))
            print('')

        #test zone
        ModelMode = False

        test_accu_cul_local = []
        test_pred_cul_local = []
        test_y_cul_local = []

        len_x_test = len(x_test)
        test_n_batch = int(len_x_test/batch_size)
        test_indx = np.random.choice(test_n_batch, len_x_test, replace=True)
        test_indx_count = dict(collections.Counter(test_indx))

        for ib in range(test_n_batch):
            test_fd = batch_distributor(ib, test_indx, x_test, y_test, x_test_length)
            test_pd, test_accu = session.run([pred, accuracy], feed_dict=test_fd)
            #generate diagnostic information
            test_accu_cul_local += [test_accu*test_indx_count[ib]]
            test_ytrue = test_fd[y]
            if ib == 0:
                test_pred_cul_local = test_pd
                test_y_cul_local = test_fd[y]
            else:
                test_pred_cul_local = np.concatenate((test_pred_cul_local, test_pd))
                test_y_cul_local = np.concatenate((test_y_cul_local, test_fd[y]))

        test_accu_cul += [sum(test_accu_cul_local)]
        x_test_len_cul += [len_x_test]
        _, _, abt = accu_by_type(test_y_cul_local, test_pred_cul_local)
        test_accu_by_type_cul += [abt]

        if ep % 50 == 0:
            print('CV {}'.format(k))
            print('epoch {}'.format(ep))
            print(' Test accuracy{}'.format(sum(test_accu_cul_local)/len_x_test))
            print(' Test accuracy by type{}'.format(abt))
            print('')

    print("Optimization Finished!")
    print(' ')

#generate output
t1 = np.array(test_accu_cul)
t2 = np.array(x_test_len_cul)

time_end = time.time()
#Diagnostic Tables:
accu_cv = t1/t2

np.savetxt('./Results/DNN/'+args.Class+'/'+args.Class+'_CV_'+str(k)+'_TestAccuracy_'+str(round(accu_cv[-1],4))+'_.txt',accu_cv,fmt='%0.6f')