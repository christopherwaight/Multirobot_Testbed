%% Circle Plotter

% Load the file matlab_circles first
% 
% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');


plot(circle30(:,1),circle30(:,2),'DisplayName','r = 30');
plot(circle35(:,1),circle35(:,2),'DisplayName','r = 35');
plot(circle45(:,1),circle45(:,2),'DisplayName','r = 45');
%plot(circle_45_3pi4(:,1),circle_45_3pi4(:,2),'DisplayName','r = 45');
% Create labels
ylabel({'Position in Y'});
xlabel({'Position in X'});

% Create title
title({'Paths traversed by 1 robot with different starating distances (r) from origin'});

box(axes1,'on');
hold(axes1,'off');
% Create legend
legend(axes1,'show');
