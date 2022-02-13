function [P,T,numberDensity] = SA76(zkm)
% [P,T,numberDensity] = SA76(zkm)
% Returns the pressure [Pa], T [K], numberDensity [m^-3] for the
% Standard Atmosphere 1976 for 0 <= zkm <= 86 km (Geometric).
%
    M = 28.9644; % Average molecular weight for air
    g0 = 9.80665; % m/s^2 acceleration due to gravity
    RE = 6378.14; % Earth's radius [km]
    T0 = 288.15; % 15C
    P0 = 101325.0; % 1 atmospere [Pa]
    R = 8.31447; % Gas Constant [J/K/mol]
    kB = 1.38065e-23; % [J/K] Boltzmann's Constant

    n = length(zkm);
    P = zeros(size(zkm));
    T = zeros(size(zkm));
    numberDensity = zeros(size(zkm));
    % Geopotential Heights
    hTbl=[0.0 11.0 20.0 32.0 47.0 51.0 71.0 RE*86.0/(RE+86.0)];
    %Temperature gradient in each Layer
    dtdhTbl=[-6.5 0.0 1.0 2.8 0.0 -2.8 -2.0];
    %Temperature Table
    tempTbl = [288.15 216.65 216.65 228.65 270.65 270.65 214.65 186.938];
    %Pressure Table
    pressureTbl = [101325.0 22632.7 5475.18 868.094 110.92 66.9478 3.95715 0.373207];


    for k=1:n
        h = zkm(k)*RE/(zkm(k)+RE);
        %Find the layer
        i = sum(hTbl <= h);
        T(k) = tempTbl(i)+dtdhTbl(i)*(h-hTbl(i));
        if(abs(dtdhTbl(i)) <=0.001)
            ratio = exp(-M*g0*(h-hTbl(i))/(R*tempTbl(i)));
        else
            ratio = ((tempTbl(i)+dtdhTbl(i)*(h-hTbl(i)))/tempTbl(i))^(-M*g0/(R*dtdhTbl(i)));
        end
        P(k) = ratio*pressureTbl(i);
    end
    numberDensity = P./(kB*T);
return