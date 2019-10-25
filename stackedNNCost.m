function [ cost, grad ] = stackedNNCost(pp,netconfig,batch_x,batch_y,nn)


stack = params2stack(pp(1:end), netconfig);
nn.W{1} = [stack{1}.b stack{1}.w];
nn.W{2} = [stack{2}.b stack{2}.w];
nn.W{3} = [stack{3}.b stack{3}.w];

stackgrad = cell(size(stack));

nn = nnff(nn, batch_x, batch_y);
nn = nnbp(nn);
nn = nnapplygrads(nn);     

cost = -(1. / nn.m) * sum(sum(batch_y .* log(nn.a{4}))) +...           %代价函数
          (nn.weightPenaltyL2 / 2.) * sum(sum(nn.W{3}.^2))+...
          (nn.weightPenaltyL2 / 2.) * sum(sum(nn.W{2}.^2))+...
          (nn.weightPenaltyL2 / 2.) * sum(sum(nn.W{1}.^2));

stackgrad{1}.w = nn.dW{1}(:,2:end);
stackgrad{1}.b = nn.dW{1}(:,1);
stackgrad{2}.w = nn.dW{2}(:,2:end);
stackgrad{2}.b = nn.dW{2}(:,1);
stackgrad{3}.w = nn.dW{3}(:,2:end);
stackgrad{3}.b = nn.dW{3}(:,1);
                 
grad = [stack2params(stackgrad)];     %梯度返回值
