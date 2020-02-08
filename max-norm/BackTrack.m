function [l] = BackTrack(P,X,Y,Omega,d_emb1,A0,A1,gamma,alpha,miu)
l=1;
while Object(P,X,Y,(A0+gamma^l*(A1-A0)),d_emb1,Omega,miu)>Object(P,X,Y,A0,d_emb1,Omega,miu)-alpha*gamma^l*trace((A0-A1)*(A0-A1)')
    l=l+1;
    if l>1000
        l=1;
        break;
    end
end
end