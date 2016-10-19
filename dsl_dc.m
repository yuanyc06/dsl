function dsl_dcn(varargin)

opts.expDir = fullfile('result', '3_DC');
opts.dataDir = 'image';
opts.gpus = [];
opts.spDir = fullfile('image', 'sp');
opts.spFullDir = fullfile('image', 'spf');
opts.modelDir = 'model';
opts.targetSize = [227 227];
opts.targetLayer = 6;
opts.locFcnDir = fullfile(opts.dataDir, 'locFcn');
opts.locCcnDir = fullfile(opts.dataDir, 'locCcn');
opts = vl_argparse(opts, varargin) ;
if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end
opts.modelPath = fullfile(opts.modelDir, 'dsl_3_dc.mat');

% conduct testing
net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
targetLayer = net.getVarIndex('prediction');
net.vars(targetLayer).precious = 1;
if ~isempty(opts.gpus)
    gpuDevice(opts.gpus) ;
    net.move('gpu');
end
net.mode = 'test' ;

imFiles = dir(fullfile(opts.dataDir, '*.jpg'));
imFiles = {imFiles.name};
imNames = cell(1, length(imFiles));
for i = 1:length(imFiles)
    [~, imNames{i}, ~] = fileparts(imFiles{i});
end
imNum = length(imNames);

for i = 1:imNum
    fprintf('%s: testing image %d of %d\n', mfilename, i, imNum)
    
    imageSizeProd2 = prod(opts.targetSize);
    imageSizeProd3 = imageSizeProd2 * opts.targetLayer;
    maskMax = single(255);

    imThis = imread(fullfile(opts.dataDir, [imNames{i}, '.jpg']));
    imThis = single(imThis);
    spThis = imread(fullfile(opts.spDir, [imNames{i}, '.png']));
    locFcn = single(imread(fullfile(opts.locFcnDir, [imNames{i}, '.png'])));
    locCcn = single(imread(fullfile(opts.locCcnDir, [imNames{i}, '.png'])));
    maskEmpty = zeros(opts.targetSize, 'single');
    imSample = cat(3,imThis,locFcn,locCcn,maskEmpty);
    spNum = double(max(spThis(:)));
    im = repmat(imSample, 1, 1, 1, spNum);

    for k = 1:spNum
        maskIdx = imageSizeProd3*(k-1)+imageSizeProd2*(opts.targetLayer-1)+find(spThis==k);
        im(maskIdx) = maskMax;
    end
    if ~isempty(opts.gpus)
        im = gpuArray(im);
    end
    net.eval({'input', im}) ;
    prediction = gather(net.vars(targetLayer).value);
    prediction = vl_nnsoftmax(prediction);
    labelOut = prediction(:,:,2,:);
    spf = imread(fullfile(opts.spFullDir, [imNames{i}, '.png']));
    salMapOut = dsl_genSal(labelOut, spf);
    imwrite(salMapOut, fullfile(opts.expDir, [imNames{i}, '.png']));
end

