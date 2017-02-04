function [ImClass,cMap] = hsiGMMLandClassIni(f,Theta,labelGT,M,N,Testindex)
% Koray Kayabol
% 07.02.2014
% Multivariate Gaussian
Kmax = max(labelGT);
% Initiliaze the mixture proportions and class maps
[NGT L] = size(f);
w = zeros(NGT,Kmax);
% Initial map making
[maxProb ind] = max(w,[],2);
ind(ind<min(labelGT)) = min(labelGT);
cMap = zeros(M,N);
for k = labelGT
    ImClass(k).index = find(ind==k);
    cMap(Testindex(ImClass(k).index)) = k;
    ImClass(k).Npix = length(ImClass(k).index);
end