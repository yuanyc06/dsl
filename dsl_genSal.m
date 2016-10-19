function salMap = dsl_genSal(labels, sp)
% Generate the saliency map with input saliency estimation 
% results and superpixel map
spNum = numel(labels);
labelsTmp = (labels - min(labels)) / (max(labels) - min(labels));
labelsTmp = uint8(labelsTmp * 255);
salMap = zeros(size(sp), 'uint8');
for i = 1:spNum
    salMap(sp == i) = labelsTmp(i);
end
