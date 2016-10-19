% Delete temp files
dataDirTemp = dir(fullfile(dataDir, 'image_*'));
dataDirTemp = {dataDirTemp.name};
for i = 1:length(dataDirTemp)
    rmdir(fullfile(dataDir, dataDirTemp{i}), 's');
end
rmdir(fullfile(dataDir, 'sp'), 's');
rmdir(fullfile(dataDir, 'spf'), 's');
rmdir(fullfile(dataDir, 'mat'), 's');
delete(fullfile(dataDir, 'imdb.mat'));
