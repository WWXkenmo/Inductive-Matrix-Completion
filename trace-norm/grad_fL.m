function [L_new] = grad_fL(P,Omega,X,Y,L,R,fre_u,fre_i,scaling_factor)
d=size(P,1);
D=size(P,2);
% set the matrix R first
R_star = R'*Y';  % row is n_factor, columne is n_samples of Y
dims = size(L);
vec_L = reshape(L,1,dims(1)*dims(2));
L_new = 0;

for i = 1:1:d
    for j=1:1:D
	   if Omega(i,j)~=2
	       x_l = X(i,:)'*R_star(:,j)';
		   dims = size(x_l);
		   vec_x_l = reshape(x_l,1,dims(1)*dims(2));
		   L_new = L_new + (vec_L*vec_x_l' - P(i,j))*x_l + scaling_factor*L;
		end
	end
end
end