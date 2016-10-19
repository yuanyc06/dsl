function featLoc = dsl_getFeatLoc(data)
% calculate x and y coordinate of superpixel's bounding box, in range
% [0,1]. this generates 4 features

[m, n] = size(data.sp);
spNum = max(data.sp(:));
featLoc = zeros(4, spNum);
for i = 1:spNum
    [r, c] = find(data.sp == i);
    x0 = min(c);
    x1 = max(c);
    y0 = min(r);
    y1 = max(r);
    x0 = (x0 - 1) / (n - 1);
    x1 = (x1 - 1) / (n - 1);
    y0 = (y0 - 1) / (m - 1);
    y1 = (y1 - 1) / (m - 1);
    featLoc(:, i) = single([x0; y0; x1; y1]);
end
