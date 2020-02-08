function [W] = squash( V,beta )
d=size(V,1);
D=size(V,2);
for k=1:1:d
    n(k)=sqrt(V(k,:)*V(k,:)');
end
[a,pai]=sort(n,'descend');
q=1;
for k=1:1:d
    s(k)=sum(n(pai(1:k)));
    if(n(pai(k))>=s(k)/(k+beta))
        q=max(q,k);
    end
end
eta=s(q)/(q+beta);
W=zeros(d,D);
for k=1:1:d
    if(k<=q)
        W(pai(k),:)=eta*V(pai(k),:)/n(pai(k));
    else
        W(pai(k),:)=V(pai(k),:);
    end
end
end