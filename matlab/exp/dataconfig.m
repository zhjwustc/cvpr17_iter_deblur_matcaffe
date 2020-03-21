function Solver = dataconfig(Solver)
Solver.patchsize = 256;
Solver.batchsize = 4;
Solver.datapath = 'path to data/';
Solver.subfolderpath = {'your data1', 'your data2'}; %% can use data from multiple sources
Solver.cleanFolder = 'clean';
Solver.blurFolder = 'blur';
Solver.kernelGtFolder = 'kernelGt';
Solver.kernelNoiseFolder = 'kernelNoise';   %% use inaccurate blur kernels sometimes
if exist(fullfile(Solver.datapath, 'data.mat'),'file')
    load(fullfile(Solver.datapath, 'data.mat'));
else 
    trainCleanlst = {};
    trainBlurlst = {};
    trainkernelGtlst = {};
    trainkernelNoiselst = {};
    % reading training data
    
    count = 1;
    for i=1:length(Solver.subfolderpath)
        dir_list = dir(fullfile(Solver.datapath,Solver.subfolderpath{i},Solver.cleanFolder,'*.png'));
        num_png = length(dir_list);
        for id = 1:num_png
            trainCleanlst{count}  = fullfile(Solver.datapath,Solver.subfolderpath{i},Solver.cleanFolder,dir_list(id).name);
            trainBlurlst{count}  = fullfile(Solver.datapath,Solver.subfolderpath{i},Solver.blurFolder,dir_list(id).name);
            trainkernelGtlst{count}  = fullfile(Solver.datapath,Solver.subfolderpath{i},Solver.kernelGtFolder,dir_list(id).name(1:end-4));
            trainkernelNoiselst{count}  = fullfile(Solver.datapath,Solver.subfolderpath{i},Solver.kernelNoiseFolder,dir_list(id).name(1:end-4));
%             if(i==1)
%                 trainDataType{count} = 'BSD';
%             else
%                 trainDataType{count} = 'flickr';
%             end
            count = count + 1;
        end
    end

    data.trainCleanlst = trainCleanlst;
    data.trainBlurlst = trainBlurlst;
    data.trainkernelGtlst = trainkernelGtlst;
    data.trainkernelNoiselst = trainkernelNoiselst;
    data.trainDataType = trainDataType;
    data.train_num = length(trainCleanlst);   
    
    
    fprintf('saving data structure ...\n');
    save(fullfile(Solver.datapath, 'data.mat'), 'data');
end
Solver.data = data; 
fprintf('Done with data config, obtain %d traning images.\n',Solver.data.train_num);
end
