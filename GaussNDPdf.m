function y = GaussNDPdf(x,a,B,d)
% Koray Kayabol

% x: data
% a: mean
% B: Covariance matrix
% d: dimension

N = length(x(:,1));
dx = zeros(N,d);
for l=1:d
    dx(:,l) = (x(:,l) - a(l));
end
dxS = dx*inv(B);
dxSdx = zeros(N,1);
for n = 1:N
    dxSdx(n) = dxS(n,:)*dx(n,:)';
end
logy =  - 0.5*dxSdx;% -0.5*log(det(2*pi*B));
y = exp(logy);