function [score Z L R] = IMCmaxNorm(X,Y, P, train_pos_id, train_neg_id, factor_n, t, miu, epsilon, gamma, alpha,bs)
Omega=ones(size(P,1),size(P,2))*2;
for i = 1:size(train_pos_id,1)
    Omega(train_pos_id(i,1),train_pos_id(i,2)) = 1;
end
for i = 1:size(train_neg_id,1)
    Omega(train_neg_id(i,1),train_neg_id(i,2)) = 0;
end

d_emb1=size(X,2);
d_emb2=size(Y,2);
d=size(X,1);
D=size(Y,1);


%L=rand(d_emb1,factor_n);
L=[eye(factor_n);zeros(d_emb1-factor_n,factor_n)];
%R=rand(d_emb2,factor_n);
R=[eye(factor_n);zeros(d_emb2-factor_n,factor_n)];


diff=epsilon+1;
A0=[L;R];
while diff>epsilon
    G=grad_fA(X,Y,P,Omega,factor_n,A0,d_emb1,bs);
    A1=squash(A0-t*G,t*miu);
    l=BackTrack(P,X,Y,Omega,d_emb1,A0,A1,gamma,alpha,miu);
    A0_new=(1-gamma^l)*A0+gamma^l*A1;
    %diff=trace((A0-A0_new)'*(A0-A0_new))/trace(A0'*A0);
	diff = abs(Object(P,X,Y,A0_new,d_emb1,Omega,miu)-Object(P,X,Y,A0,d_emb1,Omega,miu))
	Object(P,X,Y,A0_new,d_emb1,Omega,miu);
    A0=A0_new;
	t = t*0.9;
end
%%return
A = A0;
L=A(1:(d_emb1),:);
R=A((d_emb1+1):(size(A,1)),:);
Z = L*R';
score = X*Z*Y';
end