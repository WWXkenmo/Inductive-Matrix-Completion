function [R_new] = grad_fR(P,Omega,X,Y,L,R,fre_u,fre_i,scaling_factor)
d=size(P,1);
D=size(P,2);
% set the matrix R first
X_star = X*L;
R_t = R'; Y_t = Y';
dims = size(R_t);
vec_R_t = reshape(R_t,1,dims(1)*dims(2));
R_new = 0;

for i = 1:1:d
    for j=1:1:D
	   if Omega(i,j)~=2
	       x_star_l = X_star(i,:)'*Y_t(:,j)';
		   dims = size(x_star_l);
		   vec_x_star_l = reshape(x_star_l,1,dims(1)*dims(2));
           R_new = R_new + (vec_R_t*vec_x_star_l' - P(i,j))*x_star_l + scaling_factor*R_t;
		end
	end
end
R_new = R_new';
end