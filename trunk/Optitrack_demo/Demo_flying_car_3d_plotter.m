%% A file for plotting 3D Data from Motive

time = simout.Time;
data = simout.Data;
figure();
plot3( data(:,1),data(:,2),-data(:,3))
grid on;
