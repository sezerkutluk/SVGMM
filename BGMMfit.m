% MATLAB implementation of the spatially-varying Gaussian mixture model
% classification algorithmfor proposed in the floowing paper:
% 
% K. Kayabol and S. Kutluk, "Bayesian classification of hyperspectral 
% images using spatially-varying Gaussian mixture model", Digital Signal 
% Processing, vol. 59, pp. 106–114, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2016, Koray Kayabol, Sezer Kutluk      % 
% <koray.kayabol@gtu.edu.tr>, <sezer.kutluk@gmail.com> %             
% All rights reserved               			       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       f       : Training data matrix. 
%       T.index : Indeces of class labels.
%       T.Npix  : Number of pixels in the classes.
%       labelGT : Label vector.
%       L       : Spectral dimension

% Output:
%       Theta   : Estimated parameters Theta.mu and Theta.Sigma
%    

function [Theta] = BGMMfit(f,T,labelGT,L)

Kmax = length(labelGT);
TN = [T.Npix];

% prior Normal-Inverse Wishart
lamb = 0.001;
alp = 0.75;
mu0 = mean(f,1);

samplemean = zeros(Kmax,L);
SS = zeros(L,L,Kmax);
s = 0;
for k=1:Kmax
    % parameter estimation
    fgt = f(s+1:s+TN(k),:);
    samplemean(k,:) = mean(fgt,1);
    dfgt = zeros(TN(k),L);
    for l=1:L
        dfgt(:,l) = fgt(:,l) - samplemean(k,l);
    end
    SS(:,:,k) = dfgt'*dfgt/TN(k);
    s = s + T(labelGT(k)).Npix; 
end
SSm = mean(SS,3);
Psii = 0.25*SSm + 0.75*diag(diag(SSm));

for k = 1:Kmax
    Theta(labelGT(k)).mu = (TN(k)*samplemean(k,:)+lamb*mu0)/(TN(k)+lamb);
    for l=1:L
        dsm(:,l) = (samplemean(:,l) - mu0(l));
    end
    SSM = dsm'*dsm;
    tt = (alp*(TN(k)+L+1)-L-1)/(1-alp);
    Theta(labelGT(k)).Sigma = alp*Psii + (1-alp)*SS(:,:,k) + ...
        (TN(k)*lamb*SSM/(TN(k)+lamb) )/(TN(k) + tt + L + 1);
end
