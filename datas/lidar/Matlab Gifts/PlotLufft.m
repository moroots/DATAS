%% Variables
%read backscatter profile, time, and altitude from file 
InFile = '20200308_Catonsville-MD_CHM160112_000.nc';
Time = ncread(InFile,'time');
alt = ncread(InFile,'range');
Braw = ncread(InFile,'beta_raw');
Profile = double(Braw);
APA =Profile; 
tt= log10(abs(Profile));

%reformat time 
timevecUTC = datevec(Time/(3600*24) + datenum(1904, 1, 1, 0, 0, 0));
timewaveUTC = datetime(timevecUTC,'InputFormat','YYYY-MM-dd HH:mm:SS');
timenumUTC = datenum(timewaveUTC);

%% Plot backscatter 
figure;
caxis=[3.5 8.5];
imagesc(timenumUTC,alt,tt,caxis); 
datetick('x','HH:MM','keeplimits')%convets x-axis from serial num to time
h=colorbar;
ylabel(h,'Log_{10} of Aerosol Backscatter','Rotation',270,'FontSize',12,'Units','inches','Position',[0.8 2.2 0]);
colormap(jet);
newMap=[0 0 0; jet];
colormap(newMap);
ylabel('Altitude (m agl)','FontSize',12) % y-axis label
sd = datestr(timewaveUTC(1,1));%,'%1d.png');
UTC = '  (UTC)';
xlabel([UTC],'FontSize',12) % x-axis label
set(gca,'YDir','normal','FontSize',12)
shading flat
camp = 'Preliminary Data';
cei = 'Lufft CHM15k - UMBC'; 
title([cei newline sd],'FontSize',12);
s = sprintf('%02.0f',timevecUTC(1,1:3));
sy = 'Lufft_Ceilometer'; 
figname = fullfile([sy s]);
%print(figname,'-dpng', '-r300')
%close 