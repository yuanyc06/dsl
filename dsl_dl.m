function dsl_dl(varargin)

opts.expDir = fullfile('result', '1_DL');
opts.dataDir = 'image';
opts.modelDir = 'model';
opts.targetSize = [384 384];
opts.nextStepSize = [227 227];
opts.gpus = [];
opts = vl_argparse(opts, varargin) ;
opts.modelPath = fullfile(opts.modelDir, 'dsl_1_dl.mat');
opts.statsPath = fullfile(opts.modelDir, 'dsl_1_dl_imstats.mat');
load(opts.statsPath);
opts.rgbMean = reshape(rgbMean, [1 1 3]) ;
if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end

% conduct testing
net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
targetLayer = net.getVarIndex('prediction');
net.vars(targetLayer).precious = 1;
if ~isempty(opts.gpus)
    gpuDevice(opts.gpus) ;
    net.move('gpu');
end
net.mode = 'test' ;

imFiles = dir(fullfile(opts.dataDir,'*.jpg'));
imFiles = {imFiles.name};
imNames = cell(1, length(imFiles));
for i = 1:length(imFiles)
    [~, imNames{i}, ~] = fileparts(imFiles{i});
end
imNum = length(imNames);
for i = 1:imNum
    fprintf('%s: testing image %d of %d\n', mfilename, i, imNum)
    im = imread(fullfile(opts.dataDir, [imNames{i}, '.jpg'])) ;
    im_ = single(im) ; % note: 255 range
    im_ = bsxfun(@minus, im_, opts.rgbMean);
    if ~isempty(opts.gpus)
        im_ = gpuArray(im_);
    end
    net.eval({'input', im_}) ;
    prediction = gather(net.vars(targetLayer).value);
    prediction = vl_nnsoftmax(prediction);
    labelOut = prediction(:,:,2,:);
    for j = 1:size(labelOut,4)
        maskOut = labelOut(:,:,:,j);
        maskOut = imresize(maskOut, opts.nextStepSize);
        maskOut = (maskOut - min(maskOut(:))) / (max(maskOut(:)) - min(maskOut(:)));
        maskOut = uint8(255*(maskOut));
        maskName = fullfile(opts.expDir, [imNames{i}, '.png']);
        imwrite(maskOut, maskName);     
    end
end
