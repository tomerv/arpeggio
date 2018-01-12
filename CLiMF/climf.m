
[M,N]=size(Traindata);

[fi,fj,fv]=find(Traindata);
indrated=cell(M, 1);
fprintf('Indexing %d ratings...\n', size(fi, 1));
for i=1:size(fi)
    indrated{fi(i)} = [];
end
for i=1:size(fi)
    if mod(i,1000000) == 0
        fprintf('  %d...\n', i);
    end
    indrated{fi(i)} = [indrated{fi(i)} fj(i)];
end



dim=10; % num of latent factors
numiter=25; % num of iterations
lambda=0.001; % regularization coefficient
gamma=0.0001; % learning rate
U=0.01*rand(M,dim); % initialize U
V=0.01*rand(N,dim); % initialize V

% testing users in iteration
numtest=min(M, 10000); % at most 10000 users are used for testing through iteration
randid=randperm(M);
testid=randid(1:numtest);


t=0;
while t<numiter
    fprintf('iteration %d\n', t);
    for m=1:M
        if ~isempty(indrated{m})
            tempu=zeros(length(indrated{m}),dim);
            for i=1:length(indrated{m})
                temp=U(m,:)*(repmat(V(indrated{m}(i),:),[length(indrated{m}) 1])-V(indrated{m},:))';
                tempdot=logfd(temp).*(1./(1-logf(-temp))-1./(1-logf(temp)));
                V(indrated{m}(i),:)=V(indrated{m}(i),:)+gamma*((logf(-U(m,:)*V(indrated{m}(i),:)'))+sum(tempdot)*U(m,:)-lambda*V(indrated{m}(i),:));
                tempu(i,:)=logf(-U(m,:)*V(indrated{m}(i),:)')*V(indrated{m}(i),:)+(logfd(-temp)./(1-logf(-temp)))*(repmat(V(indrated{m}(i),:),[length(indrated{m}) 1])-V(indrated{m},:));
            end
            if length(indrated{m})>1
                U(m,:)=U(m,:)+gamma*(sum(tempu)-lambda*U(m,:));
            else
                U(m,:)=U(m,:)+gamma*(tempu-lambda*U(m,:));
            end
        end
    end
    t=t+1;
    pred=U(testid,:)*V';
    [rr,ftest]=mrr_at_k_metric(pred,Testdata(testid,:),10,0.000001);
    fprintf(1,'%s %d %s %8.4f\n','iteration=',t,', MRR= ',ftest);
    clear pred
end
