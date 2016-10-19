function imdb = dsl_dataSetup(varargin)

opts.dataDir = 'image';
opts.modelDir = 'model';
opts.DLSize = [384 384];
opts.DCSize = [227 227];
opts.localSize = [28, 28];
opts.proxSize = [100 100];
opts.histBinNum = 32;
opts.numSp = 200;
opts.gpus = [];
opts.imdbPath = fullfile('image', 'imdb.mat');
opts = vl_argparse(opts, varargin) ;

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    return;
end

% List images
imNamesFull = dir(fullfile(opts.dataDir, '*.jpg'));
imNamesFull = {imNamesFull.name};
imNames = cell(1, length(imNamesFull));
for i = 1:length(imNamesFull)
    [~, imNames{i}, ~] = fileparts(imNamesFull{i});
end
imNum = length(imNames);

% Data preparation
dataDirDL = fullfile(opts.dataDir, ['image_', num2str(opts.DLSize(1))]);
dataDirDC = fullfile(opts.dataDir, ['image_', num2str(opts.DCSize(1))]);
spDir = fullfile(opts.dataDir, 'sp');
spFullDir = fullfile(opts.dataDir, 'spf');
matDir = fullfile(opts.dataDir, 'mat');
if ~exist(dataDirDL, 'dir')
    mkdir(dataDirDL);
end
if ~exist(dataDirDC, 'dir')
    mkdir(dataDirDC);
end
if ~exist(spDir, 'dir')
    mkdir(spDir);
end
if ~exist(spFullDir, 'dir')
    mkdir(spFullDir);
end
if ~exist(matDir, 'dir')
    mkdir(matDir);
end

feat.global = dsl_getFeatGlobal(opts.dataDir, opts.gpus);

for i = 1:imNum    
    fprintf('%s: preparing image data %d of %d\n', mfilename, i, imNum)
    
    % Data preparation for DL and DC
    imThisFull = imread(fullfile(opts.dataDir, [imNames{i}, '.jpg']));
    [mOri, nOri, ~] = size(imThisFull);
    [imThis, wid] = removeFrame(fullfile(opts.dataDir, [imNames{i}, '.jpg']));
    imThis = uint8(255*imThis);
    imDL = imresize(imThis, opts.DLSize);
    imDC = imresize(imThis, opts.DCSize);
    imThisBmp = fullfile(opts.dataDir, [imNames{i}, '.bmp']);
    comm = [fullfile(pwd,'support','SLIC.exe'),' ',imThisBmp,' ',int2str(20),' ',int2str(opts.numSp),' '];
    if ispc
        evalc('system(comm);');
    elseif isunix
        comm = ['wine ',comm];
        evalc('unix(comm);');
    else
        error('Only Windows and Linux systems are currently supported.');
    end
    spThisName = fullfile(opts.dataDir, [imNames{i}, '.dat']);
    spThis = readDat([wid(4)-wid(3)+1,wid(6)-wid(5)+1], spThisName);
    spNum = max(spThis(:));
    delete(imThisBmp);
    delete(spThisName);
    delete([spThisName(1:end-4) '_SLIC.bmp']);
    spFullThis = zeros(mOri, nOri);
    spFullThis(wid(3):wid(4), wid(5):wid(6)) = spThis;
    spDC = imresize(spThis, opts.DCSize, 'nearest');
    
    imwrite(uint8(imDL), fullfile(dataDirDL, [imNames{i}, '.jpg']));
    imwrite(uint8(imDC), fullfile(dataDirDC, [imNames{i}, '.jpg']));
    imwrite(uint8(spFullThis), fullfile(spFullDir, [imNames{i}, '.png']));
    imwrite(uint8(spDC), fullfile(spDir, [imNames{i}, '.png']));
    
    % Data preparation for SL
    
    dataSL.global.rgb = uint8(imThis);
    dataSL.global.lab = rgb2lab(dataSL.global.rgb);
    dataSL.global.sp = spThis;
    dataSL.local.lab = zeros([opts.localSize, 3, spNum], 'single');
    dataSL.local.rgb = zeros([opts.localSize, 3, spNum], 'single');
    dataSL.prox.lab = zeros([opts.proxSize, 3, spNum], 'single');
    dataSL.prox.rgb = zeros([opts.proxSize, 3, spNum], 'single');
    for s = 1:spNum
        [r, c] = find(dataSL.global.sp == s);
        localSample.lab = dataSL.global.lab(min(r):max(r), min(c):max(c), :);
        localSample.lab = imresize(localSample.lab, opts.localSize);
        dataSL.local.lab(:,:,:,s) = localSample.lab;
        localSample.rgb = dataSL.global.rgb(min(r):max(r), min(c):max(c), :);
        localSample.rgb = imresize(localSample.rgb, opts.localSize);
        dataSL.local.rgb(:,:,:,s) = localSample.rgb;
        % get proximal data
        se = strel('diamon', 1);
        spSet = s;
        spMask = ismember(dataSL.global.sp, spSet);
        spMaskDilate = imdilate(spMask, se);
        adj1 = unique(dataSL.global.sp(xor(spMask, spMaskDilate)));
        spSet = union(spSet, adj1);
        spMask = ismember(dataSL.global.sp, spSet);
        spMaskDilate = imdilate(spMask, se);
        adj2 = unique(dataSL.global.sp(xor(spMask, spMaskDilate)));
        spSet = union(spSet, adj2);
        [r, c] = find(ismember(dataSL.global.sp, spSet));
        proxSample.lab = dataSL.global.lab(min(r):max(r), min(c):max(c), :);
        proxSample.lab = imresize(proxSample.lab, opts.proxSize);
        dataSL.prox.lab(:,:,:,s) = proxSample.lab;
        proxSample.rgb = dataSL.global.rgb(min(r):max(r), min(c):max(c), :);
        proxSample.rgb = imresize(proxSample.rgb, opts.proxSize);
        dataSL.prox.rgb(:,:,:,s) = proxSample.rgb;   
    end
    
    feat.local.color = dsl_getFeatColor(dataSL.local, opts.histBinNum);
    feat.local.location = dsl_getFeatLoc(dataSL.global);
    feat.local.conv = dsl_getFeatConv(dataSL.local, opts.gpus);
    feat.prox.color = dsl_getFeatColor(dataSL.prox, opts.histBinNum);
    
    featAll = [feat.local.color; feat.local.location; feat.local.conv; ...
        feat.prox.color; repmat(feat.global(:,i), 1, spNum)];
    
    save(fullfile(matDir, [imNames{i}, '.mat']), 'featAll');
end

imdb.dataDirDL = dataDirDL;
imdb.dataDirDC = dataDirDC;
imdb.spDir = spDir;
imdb.spFullDir = spFullDir;
imdb.matDir = matDir;

save(opts.imdbPath, '-struct', 'imdb') ;