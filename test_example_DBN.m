%function test_example_DBN
clc;
clear all; 
close all;  

load textMatrixRB10;
train_x = tfidf(train_x);
test_x = tfidf(test_x);
train_x = full(train_x);
test_x = full(test_x);


%%  train DBN and initialize NN
rand('state',0)
%train dbn
dbn.sizes = [700 100];
opts.numepochs =   10;
opts.batchsize = 400;
opts.momentum  =   0.9;
opts.alpha     =  0.001;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

nn = dbnunfoldtonn(dbn, 10);   
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  100;
opts.batchsize = 400;
[nn, L, er,bad] = nntrain(nn, train_x, train_y, opts,test_x,test_y); %er is the error rate




