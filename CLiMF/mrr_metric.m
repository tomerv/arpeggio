function [rr,f] = mrr_metric(pred,Testdata,relval)

% rr : reciprocal rank for each query
% f: Mean reciprocal rank
% pred: prediction matrix
% Testdata: Groundtruth matrix
% relval: The threshold for relevance

Nq = length(Testdata(:,1));
rr = zeros(Nq,1);
for i=1:Nq
    ind = find (Testdata(i,:)>=relval);
    if ~isempty(ind)
        [val,nb]=sort(full(pred(i,:)),'descend');
        k=1;
        while Testdata(i,nb(k))<relval
            k=k+1;
        end
        rr(i)=1/k;
    else
        rr(i)=0;
    end
end
if any(rr>0)
    f=mean(rr(rr>0));
else
    f=0;
end