clear
% EP dataset

% load training set
UPL=5;
filename=['EP25_UPL' num2str(UPL) '.mat'];
load(filename);
[M,N]=size(Traindata);
% load index of rated items for each user
filename=['EP25_UPL' num2str(UPL) '_indrated.mat'];
load(filename);

dim=10; % num of latent factors
numiter=25; % num of iterations
lambda=0.001; % regularization coefficient
gamma=0.0001; % learning rate
U=0.01*rand(M,dim); % initialize U
V=0.01*rand(N,dim); % initialize V

% testing users in iteration
numtest=100; % suppose 100 users are used for testing through iteration
randid=randperm(M);
testid=randid(1:numtest);

t=0;
while t<numiter
    
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
    [rr,ftrain]=mrr_metric(pred,Traindata(testid,:),1);
    fprintf(1,'%s %d %s %8.4f\n','iteration=',t,', MRR= ',ftrain);
    clear pred
end

%quit


