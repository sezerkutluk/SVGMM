% MATLAB demo of the spatially-varying Gaussian mixture model
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

close all;clear all;clc
% Read data
load 'Indian_pines_corrected.mat'
load 'Indian_pines_gt.mat'
im3 = indian_pines_corrected;
im3 = im3(:,:,[1:200]);
cMapGT = indian_pines_gt;
labelGT = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ];
[M,N,LL] = size(im3);

% Obtain the groundtruth indices on the image
NGT = length(find(cMapGT));
GTindex = zeros(1,NGT);
st = 0;
for k=1:length(labelGT)
    ImGT(labelGT(k)).index = find(cMapGT==labelGT(k));
    ImGT(labelGT(k)).Npix = length(ImGT(labelGT(k)).index);
    GTindex(st+1:st+ImGT(labelGT(k)).Npix) = ImGT(labelGT(k)).index;
    st = st + ImGT(labelGT(k)).Npix;
end

% Ordering HSI w.r.t. GT labels
NN = M*N; %number of pixel in image
[GTsub1 GTsub2] = ind2sub([M N],GTindex);
ford = zeros(NGT,LL);
for i=1:NGT
    ford(i,:) = im3(GTsub1(i),GTsub2(i),1:LL);
end
% Mean extraction
meanf = mean(ford,1);
for i=1:NGT
    ford(i,:) = ford(i,:) - meanf;
end

L = LL;
fd = ford;

% Divide the GT into training and test sets randomly
Kc = nnz(labelGT);
Kmax = length(labelGT);
GTN = [ImGT.Npix];
% Choose number of of training samples per class
Nper = 10*ones(Kmax,1);
% Set the Nper to min(Nper, GTN/2)
for i=1:length(Nper)
    if Nper(i) > GTN(i) / 2
        Nper(i) = GTN(i) / 2;
    end
end

Ntrain = sum(Nper);
ftrain = zeros(Ntrain,LL);
strain = 0;
s = 0;
for k=1:length(labelGT)
    ImTrain(labelGT(k)).Npix = Nper(labelGT(k));
    RndInd = randperm(ImGT(labelGT(k)).Npix);
    ImTrain(labelGT(k)).index = ImGT(labelGT(k)).index(RndInd(1:ImTrain(labelGT(k)).Npix));
    fdk = fd(s+1:s+ImGT(labelGT(k)).Npix,:);
    ftrain(strain+1:strain+ImTrain(labelGT(k)).Npix,:) = fdk(RndInd(1:ImTrain(labelGT(k)).Npix),:);
    fdTrain(k).index = s + RndInd(1:ImTrain(k).Npix);
    s = s + ImGT(labelGT(k)).Npix;
    strain = strain + ImTrain(labelGT(k)).Npix;
end
clear Rnd

% Supurvised learning of the parameters of the Bayesian Gaussian
% mixture model
% *****************************************************************
Theta = BGMMfit(ftrain,ImTrain,labelGT,L);

% Test
% *********************************
% Initialization of the class map
f = fd;
Index = GTindex;
Ntot = NGT;

wa = zeros(Ntot,Kmax);
for k = 1:length(labelGT)
    wa(:,k) = GaussNDPdf(f,Theta(labelGT(k)).mu,Theta(labelGT(k)).Sigma,L);
end
% Make the prior probability of the labels of training samples be equally
% likely
for k=1:Kc
    wa(fdTrain(k).index, :) = 1 / length(labelGT);
end

sumwa = sum(wa,2);
for k=1:Kc
    wa(:,k) = wa(:,k)./sumwa;
end

% Get the initial classification map by classifying the pixels
% independently
[LandClass,cMap] = hsiGMMLandClassIni(f,Theta,labelGT,M,N,Index);

bet = 1; % smoothing parameter
Ws = 13; % Window size
Tmax = 30; % Maximum number of iterations
t = 0;
while t<Tmax
    t=t+1;
    % Update the class labels
    % ***************************
    [zeta,bet] = MnLPdfwpar3sup(cMap,labelGT,M,N,bet,Ws);
    wz(:,labelGT) = zeta(Index,labelGT);
    
    % Posterior of z
    w = wa.*wz;
    % Normalization
    sumw = sum(w,2);
    for k = 1:length(labelGT)
        w(:,k) = w(:,k)./sumw;
    end
    
    %Classify the pixels
    [maxProb Lab] = max(w,[],2);
    Lab(Lab<min(labelGT)) = min(labelGT);
    ChangeNum(t) = 0; TNpix=0;
    for k = labelGT
        LandClass(k).index = find(Lab==k);
        LandClass(k).Npix = length(LandClass(k).index);
        cMap(Index(LandClass(k).index)) = k;
    end
end
figure
imagesc(cMap)