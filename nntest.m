function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    
    %% ��������ָ��recall(��ȫ��)��precision(��׼��)
       %�ο��������ı�����������ı��ھ򼼱��о���ʵ�֡������е������㷨һ��

   
    evaluae = zeros(3,10);  %evaluae��ÿһ�зֱ𱣴����a��b��c
    for j=1:10
        for i=1:size(expected,1)
            if (labels(i)==j) && (expected(i)==j)
                evaluae(1,j)=evaluae(1,j)+1;
            elseif (labels(i)==j) && (expected(i)~=j) 
                evaluae(2,j)=evaluae(2,j)+1;
            elseif (labels(i)~=j) && (expected(i)==j)
                evaluae(3,j)=evaluae(3,j)+1;
            end
        end
    end
    
    recall = evaluae(1,:)./(evaluae(1,:)+evaluae(3,:));
    precision = evaluae(1,:)./(evaluae(1,:)+evaluae(2,:));
    F1 = (2*recall.*precision)./(recall+precision);
    
    
end
