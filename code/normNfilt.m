clc;clear;
name={'cjc','qst','smj'};
for i=1:3
    for j=1:10
        for k=1:10
            address=['D:\2019summer\data\gesture\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
            data=importdata(address);
            data=data';
            data=data(:);
            data=medfilt1(data,63);
            offset=mean(data(1:128));
            data=data-offset;
            data=reshape(data,2048,[]);
            data=data';
            save_address=['D:\2019summer\data\gesture_normNfilt\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
            dlmwrite(save_address,data);
        end
    end
end
%% test
% i=1;
% j=1;
% k=1;
% address=['D:\2019summer\data\gesture\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
% data=importdata(address);
% data=reshape(data',1,[]);
% plot(data);
% data=medfilt1(data,63);
% offset=mean(data(1:128));
% data=data-offset;
% orgdata=data;
% data=reshape(data,2048,[]);
% data=data';
% save_address=['D:\2019summer\data\gesture_normNfilt\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
% dlmwrite(save_address,data);
