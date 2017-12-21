function [rr,f] = mrr_at_k_metric(pred,Testdata,K,relval)

% rr : reciprocal rank for each query
% f: Mean reciprocal rank
% pred: prediction matrix
% Testdata: Groundtruth matrix
% relval: The threshold for relevance

disp('starting mrr calculation');

Nq = length(Testdata(:,1));
rr = zeros(Nq,1);
usercount=0;
for i=1:Nq
    if mod(i, 1000) == 0
        fprintf('  i = %d...\n', i);
    end
    ind = find (Testdata(i,:)>=relval, 1);
    if ~isempty(ind)
        usercount=usercount+1;
        [~,nb]=sort(full(pred(i,:)),'descend');
        k=1;
        while Testdata(i,nb(k))<relval
            k=k+1;
        end
        if k <= K
            rr(i)=1/k;
        end
    end
end

if usercount>0
    f=sum(rr)/usercount;
else
    f=0;
end

