function Width = getScalarFieldWidth(ScalarFieldSelection)
%given a scalar field selection control the plotted width
%note that all plots will be square and have a size of 2*Width+1 by
%2*Width+1
%Inputs: 
%   ScalarFieldSelection: the desired scalar field to be used
%Outputs: 
%   Width: the width in m of the scalar field to be rendered. Note that
%   this is just a width not the resolution of the field to be plotted
switch ScalarFieldSelection
    case 1
        Width = 150;
    case 4
        %Testbed Sink
        Width = 5; 
    case 5
        %Wide Ridge
        %Width = 1500; 
        Width = 9000;
        %Width=300;
    case 6
        Width = 20; 
    case 7 
%         Width.ymin = -30000;
%         Width.ymax = 30000;
%         Width.xmin = 1.2E5;
%         Width.xmax = 1.7E5;
        Width.ymin = -5000;
        Width.ymax = 20000;
        Width.xmin = 1.45E5;
        Width.xmax = 1.7E5;
        Width.xwidth = (Width.xmin - Width.xmax)/2; 
        Width.ywidth = (Width.ymin - Width.ymax)/2;
        Width.xc = (Width.xmin + Width.xmax)/2; 
        Width.yc = (Width.ymin + Width.ymax)/2;
    case 8
        Width.ymin = 0;
        Width.ymax = 4.8407e+03;
        Width.xmin = 0;
        Width.xmax = 8.7401e+03;
        Width.xwidth = (Width.xmin - Width.xmax)/2; 
        Width.ywidth = (Width.ymin - Width.ymax)/2;
        Width.xc = (Width.xmin + Width.xmax)/2; 
        Width.yc = (Width.ymin + Width.ymax)/2;
    otherwise
        % case 1 composite field
        % case 2 single source
        % case 3 single sink
        Width = 300;
end
end