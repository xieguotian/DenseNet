from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    batch_norm_torch = L.BatchNormTorch(bottom, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                         scale_param=dict(bias_term=True,filler=dict(value=1),bias_filler=dict(value=0)))
                                        # filler=dict(value=1),
                                        # bias_filler=dict(value=0))
    #scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(batch_norm_torch, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                         param=[dict(lr_mult=1, decay_mult=1)],
                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    if dropout>0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv

def add_layer(bottom, num_filter, dropout):
    conv1x1 = bn_relu_conv(bottom, ks=1, nout=4*num_filter, stride=1, pad=0, dropout=dropout)
    conv3x3 = bn_relu_conv(conv1x1, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv3x3, axis=1)
    return concate

def transition(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return pooling

#change the line below to experiment with different setting
#depth -- must be 3n+4
#first_output -- #channels before entering the first dense block, set it to be comparable to growth_rate
#growth_rate -- growth rate
#dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(data_file, mode='train', batch_size=64, depth=[6,12,24,16], first_output=64, growth_rate=32, dropout=0):
    data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
              transform_param=dict(mean_file="/home/zl499/caffe/examples/cifar10/mean.binaryproto"))

    nchannels = first_output
    model = L.Convolution(data, kernel_size=7, stride=2, num_output=nchannels,
                          param=[dict(lr_mult=1, decay_mult=1)],
                        pad=3, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    model = L.BatchNormTorch(model, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                        scale_param=dict(bias_term=True, filler=dict(value=1),
                                                         bias_filler=dict(value=0)))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.MAX, kernel_size=3, stride=2,pad=1,ceil_mode=False)  # global_pooling=True)

    #N = (depth-4)/3
    for i in range(depth[0]):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    nchannels /= 2
    model = transition(model, nchannels, dropout)

    for i in range(depth[1]):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    nchannels /= 2
    model = transition(model, nchannels, dropout)

    for i in range(depth[2]):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    nchannels /= 2
    model = transition(model, nchannels, dropout)

    for i in range(depth[3]):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    #model = transition(model, nchannels, dropout)

    #model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    # model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.BatchNormTorch(model, in_place=False,
                                        param=[dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=0, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0),
                                               dict(lr_mult=1, decay_mult=0)],
                                        scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
                                        # bias_term=True,
                                        # filler=dict(value=1),
                                        # bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.AVE, kernel_size=7, stride=1)#global_pooling=True)
    model = L.InnerProduct(model, num_output=1000, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'),
                           param=[dict(lr_mult=1, decay_mult=1),
                                  dict(lr_mult=1, decay_mult=0)])
    # model = L.Convolution(model, num_output=1000, bias_term=True, weight_filler=dict(type='msra'),kernel_size=1, stride=1,pad=0,
    #                       param=[dict(lr_mult=1, decay_mult=1),
    #                              dict(lr_mult=1, decay_mult=0)],
    #                        bias_filler=dict(type='constant'))
    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)
    return to_proto(loss, accuracy)

def make_net():

    # with open('DesNet121.prototxt', 'w') as f:
    #     #change the path to your data. If it's not lmdb format, also change first line of densenet() function
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64)), file=f)
    with open('DesNet161.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
        print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64,depth=[6,12,36,24], growth_rate=48,first_output=96)), file=f)

    # with open('test_densenet.prototxt', 'w') as f:
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_test_lmdb', batch_size=50)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_densenet.prototxt'
    s.test_net.append('test_densenet.prototxt')
    s.test_interval = 800
    s.test_iter.append(200)

    s.max_iter = 230000
    s.type = 'Nesterov'
    s.display = 1

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net()
    #make_solver()










