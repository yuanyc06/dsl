function dsl_sl(varargin)

opts.expDir = fullfile('result', '2_SL');
opts.dataDir = 'image';
opts.spDir = fullfile('image', 'sp');
opts.modelDir = 'model';
opts.nextStepSize = [227 227];
opts.gpus = [];
opts = vl_argparse(opts, varargin) ;
if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end
opts.modelPath = fullfile(opts.modelDir, 'dsl_2_sl.mat');

% conduct the test
net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
targetLayer = net.getVarIndex('prediction');
net.vars(targetLayer).precious = 1;
if ~isempty(opts.gpus)
    gpuDevice(opts.gpus) ;
    net.move('gpu');
end
net.mode = 'test' ;

imFiles = dir(fullfile(opts.dataDir, '*.mat'));
imFiles = {imFiles.name};
imNames = cell(1, length(imFiles));
for i = 1:length(imFiles)
    [~, imNames{i}, ~] = fileparts(imFiles{i});
end
imNum = length(imNames);
for i = 1:imNum
    fprintf('%s: testing image %d of %d\n', mfilename, i, imNum)
    feat = load(fullfile(opts.dataDir, [imNames{i}, '.mat'])) ;
    feat = feat.featAll;
    feat = reshape(feat, [1 1 size(feat)]);
    if ~isempty(opts.gpus)
        feat = gpuArray(feat);
    end
    net.eval({'input', feat}) ;
    prediction = gather(net.vars(targetLayer).value);
    prediction = vl_nnsoftmax(prediction);
    labelOut = prediction(:,:,2,:);
    spName = fullfile(opts.spDir, [imNames{i}, '.png']);
    sp = imread(spName);
    salMapOut = dsl_genSal(labelOut, sp);
    imwrite(salMapOut, fullfile(opts.expDir, [imNames{i}, '.png']));
end
