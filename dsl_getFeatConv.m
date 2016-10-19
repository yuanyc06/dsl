function featConv = dsl_getFeatConv(data, gpus)
% generate convnet features for the superpixels of current image
% setup parameters
modelPath = 'model/dsl_2_sl_local.mat' ;
useGpu = ~isempty(gpus);

net = dagnn.DagNN.loadobj(load(modelPath)) ;
net.mode = 'test' ;
if useGpu
    net.move('gpu');
end
targetLayer = length(net.vars) - 4;
net.vars(targetLayer).precious = 1;

dataSingle = single(data.lab);
if useGpu
    dataSingle = gpuArray(dataSingle);
end
net.eval({'input', dataSingle}) ;
featConv = squeeze(gather(net.vars(targetLayer).value));
end
