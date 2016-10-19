function featGlobal = dsl_getFeatGlobal(imDir, gpus)
% generate global convnet features for all images in dataset
% setup parameters
% modelPath = 'model/imagenet-googlenet-dag.mat' ;
modelPath = 'model/dsl_2_sl_global.mat';
useGpu = ~isempty(gpus);

net = dagnn.DagNN.loadobj(load(modelPath)) ;
net.mode = 'test' ;
if useGpu
    net.move('gpu');
end
offset = 2;
targetLayer = length(net.vars) - offset;
net.vars(targetLayer).precious = 1;

imFiles = dir(fullfile(imDir,'*.jpg'));
imFiles = {imFiles.name};
imNum = length(imFiles);
featGlobal = zeros(numel(net.params(end-offset).value), imNum, 'single');
for i = 1:imNum
    im = imread(fullfile(imDir, imFiles{i})) ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
%     im_ = im_ - net.meta.normalization.averageImage ;
    imMean = reshape(net.meta.normalization.averageImage, [1 1 3]);
    im_ = bsxfun(@minus, im_, imMean);
    if useGpu
        im_ = gpuArray(im_);
    end
%     net.eval({'data', im_}) ;
    net.eval({'input', im_}) ;
    featGlobal(:,i) = squeeze(gather(net.vars(targetLayer).value));
end
