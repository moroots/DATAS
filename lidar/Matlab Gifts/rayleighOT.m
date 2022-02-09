function tau=rayleighOT(lambda,z0,z1)
% tau is the Rayleigh optical thickness [1]
% between z0 and z1 [km] at wavelength
% lambda [nm]
n = length(z1);
tau=zeros(1,n);
for i=1:n
    dz=(z1(i)-z0)./1000;
    zkm=z0:dz:z1(i);
    alpha=alphamol(lambda,zkm);
    tau(i)=sum(alpha)*dz;
end
