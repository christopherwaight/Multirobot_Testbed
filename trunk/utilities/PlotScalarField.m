%% PlotScalarField
function [s,Z] = PlotScalarField(ax,ScalarFieldSelection,resolution,DesiredValue,field_width,behavior)
% Inputs
%   ax: axes to plot on
%   ScalarFieldSelection: Simulation Scalar Field
%   Resolution: Resolution to plot field with
%   DesiredValue: Contour desired value, only used if the Contour Following
%   Behavior is on
%   field_width: width of field to plot
%   behavior: behavior that is on
%Outputs
%   s: the scalar field surface
%   Z: a 2*field_width+1 by 2*field_width+1 array of scalar field values
%   representing the scalar field readings
if ScalarFieldSelection < 7
    hold(ax,'on')
    ax.XLim = [-field_width field_width];
    ax.YLim = [-field_width field_width];
    divs = linspace(-field_width*1.5, field_width*1.5, resolution);  % 50pct extra overlap to allow for panning
    [X,Y] = meshgrid(divs,divs);

    if ScalarFieldSelection ~=5
        Z=readScalarField(X,Y,ScalarFieldSelection,[]);
    else
       %Because of the nature of the composite field calculation it cannot be
       %computed using the approach above, instead it must loop through every z
       %and y coomponent. 
        for i = 1:length(X)
            for j = 1:length(Y)
                Z(i,j) = readScalarField(X(i,j),Y(i,j),ScalarFieldSelection,[]);
            end
        end
    end
elseif ScalarFieldSelection == 7 
    hold(ax,'on')
    ax.XLim = [field_width.xmin field_width.xmax];
    ax.YLim = [field_width.ymin field_width.ymax];
    res = 100;

    x = field_width.xmin-res:res:field_width.xmax+res;
    y = field_width.ymin-res:res:field_width.ymax+res;
    
    [X,Y] = meshgrid(x,y);
    load('utilities/RealWorldData/chern_scat_int.mat','F');
    Z=readScalarField(X,Y,ScalarFieldSelection,F);
elseif ScalarFieldSelection == 8
    hold(ax,'on')
    ax.XLim = [field_width.xmin field_width.xmax];
    ax.YLim = [field_width.ymin field_width.ymax];
    res = 10;

    x = field_width.xmin-res:res:field_width.xmax+res;
    y = field_width.ymin-res:res:field_width.ymax+res;

    [X,Y] = meshgrid(x,y);
    load('utilities/RealWorldData/yos_scat_int.mat','F');
    Z=readScalarField(X,Y,ScalarFieldSelection,F);
end

s = surf(ax,X,Y,Z-1,'EdgeColor','none');  % Want surface to be slightly below robot lines
colormap('gray');
view([0 90])

% Highlight contour if specified
if (exist('DesiredValue','var') || ~isempty(DesiredValue) ) && any(strcmp(behavior,'Contour Following'))
    hold on
    [C,h] = contour3(ax,X,Y,Z,[DesiredValue DesiredValue],'ShowText','on','LineWidth',3,'LineColor',[1 0 0]);
    clabel(C,h,'FontSize',15,'Color','r','FontName','Courier')
    hold off
end
switch ScalarFieldSelection
    case 1
        p_title =sprintf('Composite Scalar Field: Field Width= %d m',2*field_width);
    case 2
        p_title =sprintf('Single Source: Field Width= %d m',2*field_width);
    case 3
        p_title = sprintf('Single Sink: Field Width= %d m',2*field_width);
    case 4
        p_title = sprintf('Testbed Single Sink: Field Width= %d m',2*field_width);
    case 5
        p_title = sprintf('Wide Ridge: Field Width=%d m',2*field_width);
    case 6
        p_title = sprintf('Front: Field Width=%d m',2*field_width);
    case 7
        p_title = sprintf('Chernobyl Scalar Field');
    case 8
        p_title = sprintf('Yosemite Scalar Field');
    otherwise
        p_title = 'Unknown Field';
end
title(p_title); 
end