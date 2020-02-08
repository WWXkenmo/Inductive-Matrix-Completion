function y = ObjectF(P,X,Y,A,d_emb1,Omega,scaling_factor)
y=0;
L=A(1:(d_emb1),:);
R=A((d_emb1+1):(size(A,1)),:);
d=size(P,1);
D=size(P,2);
for i=1:1:d
    for j=1:1:D
        if Omega(i,j)~=2
            y=y+(P(i,j)-X(i,:)*L*R'*Y(j,:)')^2;
        end
    end
end
y = y+scaling_factor*(norm(L,'fro')+norm(R,'fro'));
end