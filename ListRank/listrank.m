function [U,V]=listrank(Traindata,dim,lambda,numiter,epsilon)
% List-wise learning to rank with matrix factorization for collaborative filtering, 
% byYue Shi, Martha Larson and Alan Hanjalic
% Delft University of Technology, NL
%------------------------------------------------------------------------
% [U,V]=listrank(Traindata,dim,lambda,numiter,epsilon)
% Traindata: the training data contains user-item (row-column) ratings
% dim: the dimensionality of latent features
% lambda: the regularization parameter for the penalty on the magnitude of
% latent features
% numiter: the maximal number of iterations
% epsilon: stopping condition, e.g., 1e-6
% U: the latent user features, i.e., one user feature vector per row
% V: the latent item features, i.e., one item feature vector per row
% -------------------------------------------------------------------------

[M,N]=size(Traindata);
Traindata=Traindata/max(max(Traindata));
U=0.1*rand(M,dim);
V=0.1*rand(N,dim);
%temp = exp(logf(U*V')).*(Traindata>0);
temp = sparse_matmul(U,V',@(x)exp(logf(x)),(Traindata>0));
%tempr = exp(Traindata).*(Traindata>0);
tempr = sparse_func(Traindata, @exp);
W = sparse(1:M,1:M,1./sum(temp,2),M,M);
Wr = sparse(1:M,1:M,1./sum(tempr,2),M,M);
temp=W*temp;
tempr=Wr*tempr;
epsilon=1e-6;
E1=-sum(tempr(Traindata>0).*log(temp(Traindata>0)+epsilon))+0.5*lambda*(sum(sum(U.^2))+sum(sum(V.^2)));
t=0;
beta=1;  % learning rate
while t<numiter
    beta=beta*2;
    logfdUV=sparse_matmul(U,V',@logfd,(Traindata>0));
    %temp1=V'*((temp-tempr).*logfd(U*V').*(Traindata>0))';
    temp1=V'*((temp-tempr).*logfdUV)';
    nextU=U-beta*(temp1+lambda*U')';
    %temp2=U'*((temp-tempr).*logfd(U*V').*(Traindata>0));
    temp2=U'*((temp-tempr).*logfdUV);
    nextV=V-beta*(temp2+lambda*V')';
    %temp=exp(logf(nextU*nextV')).*(Traindata>0);
    temp=sparse_matmul(nextU, nextV', @(x) exp(logf(x)), (Traindata>0));
    W=sparse(1:M,1:M,1./sum(temp,2),M,M);
    %save(sprintf('U%d',t), 'U');
    %save(sprintf('V%d',t), 'V');
    %save(sprintf('nextU%d',t), 'nextU');
    %save(sprintf('nextV%d',t), 'nextV');
    %save(sprintf('temp%d',t), 'temp');
    %save(sprintf('W%d',t), 'W');
    if any(isinf(W(:)))
        disp('Error: saw Inf');
        break;
    end
    if any(isnan(W(:)))
        disp('Error: saw NaN');
        break;
    end
    temp=W*temp;
    E2=-sum(tempr(Traindata>0).*log(temp(Traindata>0)+epsilon))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
    while E2>E1
        beta=beta/2;
        nextU=U-beta*(temp1+lambda*U')';
        nextV=V-beta*(temp2+lambda*V')';
        %temp=exp(logf(nextU*nextV')).*(Traindata>0);
        temp=sparse_matmul(nextU, nextV', @(x) exp(logf(x)), (Traindata>0));
        W=sparse(1:M,1:M,1./sum(temp,2),M,M);
        temp=W*temp;
        E2=-sum(tempr(Traindata>0).*log(temp(Traindata>0)+epsilon))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
    end
    U=nextU;
    V=nextV;
    delta=(E1-E2)/E1;
    if delta<=epsilon
        break;
    else
        E1=E2;
        t=t+1;
        fprintf(1,'%s%d %s%8.6f %s%8.6f %s%8.6f\n','iteration=',t,'learning rate=',beta,'Obj=',E1,'deltaObj=',delta);
    end
end
