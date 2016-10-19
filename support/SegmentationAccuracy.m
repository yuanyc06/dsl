classdef SegmentationAccuracy < dagnn.Loss

  properties (Transient)
    pixelAccuracy = 0
    meanAccuracy = 0
    meanIntersectionUnion = 0
    confusion = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      predictions = gather(inputs{1}) ;
      labels = gather(inputs{2}) ;
      
      [~, predictions] =  max(predictions, [], 3);
      diffPredict = bsxfun(@ne, predictions, labels);
      obj.pixelAccuracy = sum(diffPredict(:)) / numel(diffPredict);
      obj.average = obj.pixelAccuracy;
      obj.numAveraged = obj.numAveraged + numel(labels) ;
      outputs{1} = obj.average ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.confusion = 0 ;
      obj.pixelAccuracy = 0 ;
      obj.meanAccuracy = 0 ;
      obj.meanIntersectionUnion = 0 ;
      obj.average = [0;0;0] ;
      obj.numAveraged = 0 ;
    end

    function str = toString(obj)
      str = sprintf('acc:%.2f, mAcc:%.2f, mIU:%.2f', ...
                    obj.pixelAccuracy, obj.meanAccuracy, obj.meanIntersectionUnion) ;
    end

    function obj = SegmentationAccuracy(varargin)
      obj.load(varargin) ;
    end
  end
end
