function [] = AttractProperties(data,Num_Robots)
g = ceil(sqrt(numel(data)));
f_ell = figure;
f_ellcom = figure();
p_ell = uipanel('Parent',f_ell,'BorderType','none');
p_ell.Title = 'Bounding Ellipse Overlay';
p_ell.TitlePosition = 'centertop';
p_ell.FontSize = 12;
p_ell.FontWeight = 'bold';
p_ellcom = uipanel('Parent',f_ellcom,'BorderType','none');
p_ellcom.Title = 'COM Ellipse Overlay';
p_ellcom.TitlePosition = 'centertop';
p_ellcom.FontSize = 12;
p_ellcom.FontWeight = 'bold';
for j = 1:length(data)
    % Create square grid of subplots for all parallel simulations
    P    = subplot(g,g,j,'Parent',p_ell);
    hold(P,'on')
    for i = 1:Num_Robots
        Xf(i,1) = data(1,j).robot(i).x(end);
        Xf(i,2) = data(1,j).robot(i).y(end);
        for k = 1:Num_Robots
            ds(i,k) = sqrt((data(1,j).robot(i).x(end)-data(1,j).robot(k).x(end))^2 + (data(1,j).robot(i).y(end)-data(1,j).robot(k).y(end))^2);
        end
    end
    [A , c] = MinVolEllipse(Xf', 0.001);
    if Num_Robots>2
        Ellipse_plot(A, c,P)
        scatter(P,Xf(:,1), Xf(:,2),150)
        thetas = 0:1:360;
        a_radius = 5;
        b_radius = 1;
        %scatter(sum(Xf(:,1))/Num_Robots,sum(Xf(:,2))/Num_Robots,'kx')
        %plot(a_radius*cosd(thetas)+ sum(Xf(:,1))/Num_Robots, a_radius*sind(thetas)+ sum(Xf(:,2))/Num_Robots,'-.');
        %plot(a_radius*cosd(thetas), a_radius*sind(thetas),'-.');
        % plot(b_radius*cosd(thetas)+ sum(Xf(:,1))/Num_Robots, b_radius*sind(thetas)+ sum(Xf(:,2))/Num_Robots,'m--');
        axis(P,'equal')
        box(P,'on')
        grid(P,'on')
        xlabel('X (m)');
        ylabel('Y (m)');
        
        [~, D, ~] = svd(A);
        
        % get the major and minor axes
        %------------------------------------
        as(j) = 1/sqrt(D(1,1));
        bs(j) = 1/sqrt(D(2,2));
        dmax(j) = max(max(ds));
        e = sqrt(1 - (as(j)^2)/(bs(j).^2));
        title(P,sprintf(' e = %.3g',e));
    end
    hold(P,'off')
    % Create square grid of subplots for all parallel simulations for com
    % ellipse plot
    Pcom = subplot(g,g,j,'Parent',p_ellcom);
    hold(Pcom,'on')
    scatter(Xf(:,1), Xf(:,2),150)
    a =25; %m 
    e =.9; 
    alpha = 0+45;
    b = a * sqrt(1-e^2);
    xc = sum(Xf(:,1))/Num_Robots;
    yc = sum(Xf(:,2))/Num_Robots;
    scatter(Pcom,xc,yc,'kx')
    numPoints = 100; % Less for a coarser ellipse, more for a finer resolution.
    % Make equations:
    t = linspace(0, 2 * pi, numPoints); % Absolute angle parameter
    X = a * cos(t);
    Y = b * sin(t);
    % Compute angles relative to (x1, y1).
    x = xc + X * cosd(alpha) - Y * sind(alpha);
    y = yc + X * sind(alpha) + Y * cosd(alpha);
    % Plot the ellipse as a blue curve.
    plot(Pcom,x,y,'b-', 'LineWidth', 2);	% Plot ellipse
    grid on;
    axis equal
    box on
    xlabel('X (m)');
    ylabel('Y (m)');
    
    %%Check that all robots are in the prescribed ellipse
    EQ = (((Xf(:,1)-xc)*cosd(alpha)+ (Xf(:,2)-yc)*sind(alpha)).^2 )./ (a^2) + (((Xf(:,1)-xc)*sind(alpha)- (Xf(:,2)-yc)*cosd(alpha)).^2 )./ (b^2);
    title(Pcom, sprintf('Max EQ = %0.3g', max(EQ)))
    hold(Pcom,'off')
    disp('EQs')
    disp(EQ)
    disp('Min EQs')
    disp(min(EQ)); 
    disp('Max EQs')
    disp(max(EQ)); 
end

%% Calculate max distance for simulation

if Num_Robots>2
    disp('The minor axes were:')
    disp(as)
    disp('The major axes were:')
    disp(bs)
    disp('Min and max minor axis were:')
    disp(min(as))
    disp(max(as))
    disp('Min and max major axis were:')
    disp(min(bs))
    disp(max(bs))
    disp('Average minor axis and average major axis were')
    disp(mean(as))
    disp(mean(bs))
    disp('Standard dev for minor axis and major axis were')
    disp(std(as))
    disp(std(b))
    es = sqrt(1 - (as.^2)./(bs.^2));
    disp('All eccentricities:')
    disp(es);
    disp('Min and max eccentricity')
    disp(min(es));
    disp(max(es));
    disp('Average eccentricity:')
    disp(mean(es));
    disp('Max Distance between robots')
    disp(dmax);
    disp('Max and min distances')
    disp(max(dmax));
    disp(min(dmax));
    disp('Mean distances')
    disp(mean(dmax))
end
end


