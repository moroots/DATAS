function ior = ns(lambda)
% ns(lambda) = the index of refraction of dry air at STP 
%              for wavelength lambda in nm
%
ior = 1.0 + (5791817.0./(238.0185 - (10^6)./lambda.^2)+167909.0./(57.362-(10^6)./lambda.^2)).*1.e-8;
return