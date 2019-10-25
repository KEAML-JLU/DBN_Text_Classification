function dbn = dbnsetup(dbn, x, opts)
%x��ѵ������
%������ͨ��dbn = dbnsetup(dbn, train_x, opts); �����ݵ�����
    n = size(x, 2); %��ȡ�������ݵ�ά����Ҳ����������Բ�Ľڵ���
    dbn.sizes = [n, dbn.sizes]; %dbn�ĸ���Ľڵ���������dbn.sizes��

    for u = 1 : numel(dbn.sizes) - 1    %numel(A)��ʾ���ؾ���A��Ԫ�صĸ���
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));  %��ʼ��Ȩֵ
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1); %��ʼ���Բ�ƫ��
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); %��ʼ������ƫ��
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
