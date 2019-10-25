function dbn = dbnsetup(dbn, x, opts)
%x是训练数据
%函数是通过dbn = dbnsetup(dbn, train_x, opts); 来传递的数据
    n = size(x, 2); %获取测试数据的维数，也就是输入的显层的节点数
    dbn.sizes = [n, dbn.sizes]; %dbn的各层的节点数保存在dbn.sizes中

    for u = 1 : numel(dbn.sizes) - 1    %numel(A)表示返回矩阵A中元素的个数
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));  %初始化权值
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1); %初始化显层偏置
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); %初始化隐层偏置
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
