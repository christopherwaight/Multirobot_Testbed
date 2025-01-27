function behaviorname = getBehaviorName(behavior)
behaviorname = '';
if ~isempty(behavior)
for i = 1:length(behavior)-1
    behaviorname = strcat(behaviorname,behavior{i}); 
    behaviorname = strcat(behaviorname,'-'); 
end
behaviorname= strcat(behaviorname,behavior{end});
end
end