function rs = rayleigh(lambda)
% rayleigh(lambda) = The Rayleigh scattering cross section per molecule [m^-3] for
%                 lambda in nm.
%
nstp = 2.54691e25;
rs = (1.e36*24*pi^3*(ns(lambda).^2-1).^2)./(lambda.^4.*nstp.^2.*(ns(lambda).^2+2).^2).*((6+3*rhon(lambda))./(6-7*rhon(lambda)));
return
