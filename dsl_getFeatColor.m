function featColor = dsl_getFeatColor(data, binNum)
% histogram in Lab and RGB with 32 bins respectively, plus mean, variance and
% entropy for each channel on both colorspaces. 32*3*2+6*3=210 features in total.
sampleNum = size(data.lab, 4);
dataLab1 = double(data.lab(:,:,1,:));
dataLab2 = double(data.lab(:,:,2,:));
dataLab3 = double(data.lab(:,:,3,:));
dataRgb1 = double(data.rgb(:,:,1,:));
dataRgb2 = double(data.rgb(:,:,2,:));
dataRgb3 = double(data.rgb(:,:,3,:));

for i = 1:sampleNum
    dataLab1This = reshape(dataLab1(:,:,:,i), [], 1);
    dataLab2This = reshape(dataLab2(:,:,:,i), [], 1);
    dataLab3This = reshape(dataLab3(:,:,:,i), [], 1);
    dataRgb1This = reshape(dataRgb1(:,:,:,i), [], 1);
    dataRgb2This = reshape(dataRgb2(:,:,:,i), [], 1);
    dataRgb3This = reshape(dataRgb3(:,:,:,i), [], 1);
    f = [];
    histLab1 = hist(dataLab1This, binNum)';
    histLab2 = hist(dataLab2This, binNum)';
    histLab3 = hist(dataLab3This, binNum)';
    histRgb1 = hist(dataRgb1This, binNum)';
    histRgb2 = hist(dataRgb2This, binNum)';
    histRgb3 = hist(dataRgb3This, binNum)';
    f = [f; histLab1 ./ sum(histLab1)];
    f = [f; histLab2 ./ sum(histLab2)];
    f = [f; histLab3 ./ sum(histLab3)];
    f = [f; histRgb1 ./ sum(histRgb1)];
    f = [f; histRgb2 ./ sum(histRgb2)];
    f = [f; histRgb3 ./ sum(histRgb3)];
    f = [f; mean(dataLab1This); mean(dataLab2This); ...
        mean(dataLab3This); mean(dataRgb1This); ...
        mean(dataRgb2This); mean(dataRgb3This)];
    f = [f; var(dataLab1This); var(dataLab2This); ...
        var(dataLab3This); var(dataRgb1This); ...
        var(dataRgb2This); var(dataRgb3This)];
    featColor(:,i) = single(f);
end
