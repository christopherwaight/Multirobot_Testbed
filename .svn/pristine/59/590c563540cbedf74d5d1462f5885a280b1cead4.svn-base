function [ ] = plot_path_chern( file, resolution )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

load(file);
figure
ymin = -30000;
ymax = 30000;
xmin = 1.2E5;
xmax = 1.7E5;

res = 100;

x = xmin-res:res:xmax+res;
y = ymin-res:res:ymax+res;

[X,Y] = meshgrid(x,y);

load('chern_scat_int.mat','F')
disp('Scattered Interpolant Loaded');
Z = F(X,Y);
disp('Interpolation of Meshgrid complete');

time = [0:resolution:data.Time(end)];

v = 0:5:11000;
v = [0:.5:10]*1000;
contour(X,Y,Z,v);
%heatmap(X,Y,Z);
disp('Contour Generated');
hold on

plot(interp1(data.Time, data.Data(:,9), time),interp1(data.Time, data.Data(:,10), time),'kx')
xlabel('Y (m)');
ylabel('X (m)');
  set(gca,'FontSize',10,'fontWeight','bold')
xlim([xmin, xmax])
ylim([ymin, ymax])
set(findall(gcf,'type','text'),'FontSize',10,'fontWeight','bold')
colorbar
axis equal

end