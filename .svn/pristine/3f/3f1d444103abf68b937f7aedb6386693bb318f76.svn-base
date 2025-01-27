function [] = spacingMatrixParallel(datas,SimParams)
% uses data from either a single simulation or parallel simulations and
% calculates the lattice factor (lattice spacing between robobots). It then
% produces a weighted graph only including edges that are less than 1.25
% times the avoidance range. This is to visualize the "lattice structure"
% of the swarm
% Inputs:
%   datas: an array of data output from the simulations
%   SimParams: a struct containing information about the simulations
f1 = figure;
p_1 = uipanel('Parent',f1,'BorderType','none'); 
p_1.Title = 'Lattice Plot Smallest Edges'; 
p_1.TitlePosition = 'centertop'; 
p_1.FontSize = 12;
p_1.FontWeight = 'bold';
f2 = figure;
p_2 = uipanel('Parent',f2,'BorderType','none'); 
p_2.Title = 'Lattice Plot All Edges'; 
p_2.TitlePosition = 'centertop'; 
p_2.FontSize = 12;
p_2.FontWeight = 'bold';
Ns = ceil(sqrt(length(datas)));
[~,L] = size(datas);
%Loop through each simulation data
for n = 1:L
    %extract out robot positions
    data = datas(n);
    xs = [data.robot.x];
    x = xs(end,:);
    ys = [data.robot.y];
    y = ys(end,:);
    names = {}; 
    %calculate distance between each robot pair
    for i = 1:length(x)
        for j = 1:length(y)
            d(i,j) = sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2);
        end
    end
    %build arrays for weighted graph: s and t are arrays of edge pairs
    s1 = [];
    t1 = [];
    weights1 = [];
    s2 = [];
    t2 = [];
    allWeights = [];
    for i = 1:length(x)
        for j = i:length(x)
            d = sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2);
            if 0< d && d< 1.25*SimParams.AVOID_RANGE
                s1(end+1) = i;
                t1(end+1) = j;
                weights1(end+1) = floor(d*100)/100;
            end
            s2(end+1) = i;
            t2(end+1) = j;
            allWeights(end+1) =d;
        end
        names{end+1} = num2str(i); 
    end
    latticeFactors(n) = mean(weights1);
    %produce weighted graph, note that edge weights are normalized to 1
    figure(f1)
    hold on
    G1 = graph(s1,t1,weights1./SimParams.AVOID_RANGE,length(x));
    subplot(Ns,Ns,n,'Parent',p_1)
    if length(weights1) ~= 0
        plot(G1,'XData',x,'YData',y,'EdgeLabel',G1.Edges.Weight)
    else
        scatter(x,y,'.');
    end
    box on 
    grid on 
    xlabel('X (m)');
    ylabel('Y (m)'); 
    hold off
    %produce weighted graph showing distance of all nodes
    figure(f2)
    G2 = graph(s2,t2,allWeights,names,'omitselfloops');
    subplot(Ns,Ns,n,'Parent',p_2)
    hold on
    if length(allWeights) ~= 0
        plot(G2,'XData',x,'YData',y,'EdgeLabel',G2.Edges.Weight)
    else
        scatter(x,y,'.');
    end
    A = adjacency(G2,'weighted'); 
    A = full(A); 
    xlabel('X (m)')
    ylabel('Y (m)')
    grid on
    box on
    hold off
    W_maxs(n) = max(allWeights);
end
%Display lattice factor ie spacing between robots
latticeFactor = mean(latticeFactors)
%display average max distance between two robots
W_max = mean(W_maxs)
%display minimum max distance between two robots
W_max_min = min(W_maxs)
%display maximum max distance between two robots
W_max_max = max(W_maxs)
hold off
end

