function [G] = grad_fA(X,Y,P,Omega,factor_n,A,d_emb1,bs)
L=A(1:(d_emb1),:);
R=A((d_emb1+1):(size(A,1)),:);
d=size(Omega,1);
D=size(Omega,2);
% set the matrix R first
R_star = R'*Y';  % row is n_factor, columne is n_samples of Y
dims = size(L);
vec_L = reshape(L,1,dims(1)*dims(2));
L_new = 0;

X_star = X*L;
R_t = R'; Y_t = Y';
dims = size(R_t);
vec_R_t = reshape(R_t,1,dims(1)*dims(2));
R_new = 0;

%%%Using SGD
[idx,idy] = find(Omega~=2);
batch = randperm(size(idx,1),floor(bs*size(idx,1)));
idx = idx(batch);
idy = idy(batch);
for l = 1:1:size(batch,2)
	   i = idx(l,:);
	   j = idy(l,:);
	   if Omega(i,j)~=2
	       x_l = X(i,:)'*R_star(:,j)';
		   dims = size(x_l);
		   vec_x_l = reshape(x_l,1,dims(1)*dims(2));
		   L_new = L_new + (vec_L*vec_x_l' - P(i,j))*x_l;
		   
		   x_star_l = X_star(i,:)'*Y_t(:,j)';
		   dims = size(x_star_l);
		   vec_x_star_l = reshape(x_star_l,1,dims(1)*dims(2));
           R_new = R_new + (vec_R_t*vec_x_star_l' - P(i,j))*x_star_l;
		end
	end
G = [L_new;R_new'];
end

