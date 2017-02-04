function [eta,bet] = MnLPdfwpar3sup(cpad,labelGT,mpad,npad,bet,zws)
% Koray Kayabol
% 22.01.2015
Npad = mpad*npad;
Kmax = max(labelGT);
sumfil = ones(zws,zws);
sumfil((zws+1)/2,(zws+1)/2) = 1;
D = (zws^2);

%Estimation of bet
Pcount = zeros(Npad,Kmax);
eta = zeros(Npad,Kmax);
classmask = zeros(Npad,Kmax);
for k = labelGT
    cpadbin = zeros(mpad,npad);
    ind = cpad==k;
    len(k) = length(ind);
    cpadbin(ind) = 1;
    Pcount(:,k) = reshape(filter2(sumfil,cpadbin),Npad,1)/D;
    eta(:,k) = exp(bet*(Pcount(:,k))); %Logistic
    classmask(:,k) = reshape(cpadbin,Npad,1);
end

sumjPe = sum(Pcount.*eta,2);
sumje = sum(eta,2);
sumjPPe = sum(Pcount.*Pcount.*eta,2);
sumeta = sum(eta,2);
for k = labelGT
    eta(:,k) = eta(:,k)./sumeta;
end

g0 = sumjPe./sumje;
h0 = (sumjPPe.*sumje-sumjPe.^2)./sumje.^2;
g1 = zeros(mpad*npad,1);
for k = labelGT
    g1 = g1 + classmask(:,k).*(Pcount(:,k) - g0);
end

gradbet = sum(sum(Pcount))-sum(sumjPe./sumje);
Hbet = -sum( (sumjPPe.*sumje-sumjPe.^2)./sumje.^2 );
bet = bet - 0.05*gradbet/Hbet;
bet = min(max(0,bet),100);