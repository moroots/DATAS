function pc = Pchandra(lambda,theta)
% pc = Pchandra(lambda,theta)
% This function calculates the phase function for atmospheric
% scattering with the correction due to Chandrasekhar.
% Wavelength lambda is in nm, scattering angle theta is in radians.
gamma = rhon(lambda)./(2.0-rhon(lambda));
pc = 3*((1+3.0*gamma)+((1.0-gamma).*cos(theta).^2))./(4.0*(1.0+2.0*gamma));
return