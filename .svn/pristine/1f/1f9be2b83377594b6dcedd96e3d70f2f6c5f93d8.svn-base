filename = 'chernobyl_data.csv';
M = csvread(filename,1,0);
meters_data = lla2flat([M(:,1), M(:,2), zeros(length(M(:,1)), 1)],[50,30],0,0);

F = scatteredInterpolant(meters_data(:,1),meters_data(:,2), M(:,3), 'natural'); %generates the scattered interpolant

save('chern_scat_int','F')