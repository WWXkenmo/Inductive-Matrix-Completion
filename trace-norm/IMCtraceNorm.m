function [score Z L R] = IMCmaxNorm(X,Y, P, train_pos_id, train_neg_id, factor_n, scaling_factor,step,epsilon,all_epsilon)
Omega=ones(size(P,1),size(P,2))*2;
for i = 1:size(train_pos_id,1)
    Omega(train_pos_id(i,1),train_pos_id(i,2)) = 1;
end
for i = 1:size(train_neg_id,1)
    Omega(train_neg_id(i,1),train_neg_id(i,2)) = 0;
end

Omega_filter = Omega;
Omega_filter(find(Omega_filter==2)) = 0;
S = size(Omega_filter,1)*size(Omega_filter,2);
fre_u = sum(Omega_filter,2)/S;
fre_i= sum(Omega_filter,1)/S;
fre_i = fre_i';

d_emb1=size(X,2);
d_emb2=size(Y,2);
d=size(X,1);
D=size(Y,1);


%L=rand(d_emb1,factor_n);
L=[eye(factor_n);zeros(d_emb1-factor_n,factor_n)];
%R=rand(d_emb2,factor_n);
R=[eye(factor_n);zeros(d_emb2-factor_n,factor_n)];


diff_R=epsilon+1;
diff_L=epsilon+1;
diff_all=epsilon+1;

A0=[L;R];
Omega_filter = Omega;
Omega_filter(find(Omega_filter==2)) = 0;
S = size(Omega_filter,1)*size(Omega_filter,2);
fre_u = sum(Omega_filter,2)/S;
fre_i= sum(Omega_filter,1)/S;
fre_i = fre_i';

diff_all=epsilon+1;
while diff_all>all_epsilon
    L=A0(1:(d_emb1),:);
    R=A0((d_emb1+1):(size(A0,1)),:);
    diff_R=epsilon+1;
    diff_L=epsilon+1;

        while diff_R>epsilon
            G=grad_fR(P,Omega,X,Y,L,R,fre_u,fre_i,scaling_factor);
            R_new = R-step*G;
            diff_R=trace((R-R_new)'*(R-R_new))/trace(R'*R_new);
            R=R_new;
        end
		
        while diff_L>epsilon
           G=grad_fL(P,Omega,X,Y,L,R,fre_u,fre_i,scaling_factor);
           L_new = L-step*G;
           diff_L=trace((L-L_new)'*(L-L_new))/trace(L'*L_new);
           L=L_new;
        end
    A = [L;R];
    %diff_all=trace((A0-A)'*(A0-A))/trace(A0'*A0);
	diff_all = abs(ObjectF(P,X,Y,A,d_emb1,Omega,scaling_factor)/(size([train_pos_id;train_neg_id],1)*size([train_pos_id;train_neg_id],2))-ObjectF(P,X,Y,A0,d_emb1,Omega,scaling_factor)/(size([train_pos_id;train_neg_id],1)*size([train_pos_id;train_neg_id],2)))
    A0=A;
end
%%return
A = A0;
L=A(1:(d_emb1),:);
R=A((d_emb1+1):(size(A,1)),:);
Z = L*R';
score = X*Z*Y';
end