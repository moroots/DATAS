function bm = betamol(lambda,zkm)
% bm = betamol(lambda,zkm)
% This calculates the molecular (Rayleigh) backscattering coefficient
% in [km^-1 sr^-1] with the assumption of a SA76 atmosphere.
% lambda is the wavelength in nm, and zkm is the altitude in km.
[P,T,numberDensity] = SA76(zkm);
bm = 1000*numberDensity.*Pchandra(lambda,pi).*rayleigh(lambda)/(4*pi);
return