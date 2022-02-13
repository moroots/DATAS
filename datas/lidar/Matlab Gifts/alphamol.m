function am = alphamol(lambda,zkm)
% am = alphamol(lambda,zkm)
% This calculates the molecular (Rayleigh) extinction coefficient
% in [km^-1] with the assumption of a SA76 atmosphere.
% lambda is the wavelength in nm, and zkm is the altitude in km.
[P,T,n] = SA76(zkm);
am = n.*rayleigh(lambda)*1000;
return