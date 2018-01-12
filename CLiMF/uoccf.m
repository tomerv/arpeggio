
[M,N]=size(Traindata);

indrated=cell(M, 1);
for i=1:M
    indrated{i} = find(Traindata(i,:));
end

dim=10; % num of latent factors
numiter=25; % num of iterations
lambda=0.001; % regularization coefficient
gamma=0.0001; % learning rate
rng(1234);
U=0.01*rand(M,dim); % initialize U
V=0.01*rand(N,dim); % initialize V

% testing users in iteration
numtest=min(M, 10000); % at most 10000 users are used for testing through iteration
randid=randperm(M);
testid=randid(1:numtest);

alpha=0.1;

t=0;
while t<numiter
    fprintf('iteration %d\n', t);
    for m=1:M
        
        if ~isempty(indrated{m})
            Um = U(m,:);
            tempu=zeros(length(indrated{m}),dim);
            for i=1:length(indrated{m})
                Vi = V(indrated{m}(i),:);
                temp=Um*(repmat(Vi,[length(indrated{m}) 1])-V(indrated{m},:))';
                tempdot=logfd(temp).*(1./(1-logf(-temp))-1./(1-logf(temp)));
                dLdV_climf = -(logf(-Um*Vi')) - sum(tempdot)*Um;
                UV = Um*Vi';
                dLdV_pmf = (logf(UV) - Traindata(m,indrated{m}(i))) * logfd(UV) * Um;
                dLdV_reg = Vi;
                dLdV = (1-alpha)*dLdV_climf + alpha*dLdV_pmf + lambda*dLdV_reg;
                V(indrated{m}(i),:) = Vi - gamma*dLdV;
                Vi = V(indrated{m}(i),:);
                tempu(i,:) = - logf(-Um*Vi')*Vi - (logfd(-temp)./(1-logf(-temp)))*(repmat(Vi,[length(indrated{m}) 1])-V(indrated{m},:));
                dLdU_pmf = (logf(UV) - Traindata(m,indrated{m}(i))) * logfd(UV) * Vi;
                U(m,:) = U(m,:) - gamma*alpha*dLdU_pmf;
            end
            if length(indrated{m})>1
                U(m,:) = Um - gamma*((1-alpha)*sum(tempu)+lambda*Um);
            else
                U(m,:) = Um - gamma*((1-alpha)*tempu+lambda*Um);
            end
        end
    end
    t=t+1;
    pred=U(testid,:)*V';
    [rr,ftest]=mrr_at_k_metric(pred,Testdata(testid,:),10,0.000001);
    fprintf(1,'%s %d %s %8.4f\n','iteration=',t,', MRR= ',ftest);
    clear pred
end
