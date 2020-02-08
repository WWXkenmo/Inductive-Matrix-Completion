function y = Object(P,X,Y,A,d_emb1,Omega,miu)
y=0;
L=A(1:(d_emb1),:);
R=A((d_emb1+1):(size(A,1)),:);
d=size(P,1);
D=size(P,2);
[idx,idy] = find(Omega~=2);
for l = 1:1:size(idx,1)
	   i = idx(l,:);
	   j = idy(l,:);
        if Omega(i,j)~=2
            y=y+(P(i,j)-X(i,:)*L*R'*Y(j,:)')^2;
        end
    end
y=y+miu*max(max(diag(L*L')),max(diag(R*R')));
end