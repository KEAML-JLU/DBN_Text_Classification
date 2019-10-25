function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    
    %% 计算评价指标recall(查全率)和precision(查准率)
       %参看《面向文本分类的中文文本挖掘技本研究及实现》文章中的评价算法一节

   
    evaluae = zeros(3,10);  %evaluae的每一行分别保存的是a，b，c
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
