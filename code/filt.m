clc;clear;
name={'cjc','qst','smj'};
% for i=1:3
%     for j=1:10
%         for k=1:10
%             address=['D:\2019summer\data\gesture\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
%             orgdata=importdata(address);
%             data=orgdata';
%             data=data(:);
%             data_filt=medfilt1(data,32);
%             data_filt(1)=data_filt(2);
%             data_filt=reshape(data_filt,[],2048);
%             data_filt=data_filt'; 
%             save_address=['D:\2019summer\data\gesture_filt\',name{i},'_',num2str(j),'_',num2str(k-1),'.txt'];
%             dlmwrite(save_address,data_filt)
% %             save(save_address,'data_filt');
%         end
%     end
% end
%% 
i=1;
address=['D:\2019summer\data\gesture\',name{i},'_',num2str(1),'_',num2str(0),'.txt'];
save_address=['D:\2019summer\data\gesture_filt\',name{i},'_',num2str(1),'_',num2str(0),'.txt'];
orgdata=importdata(address);
data=orgdata';
data=data(:);
filtdata=importdata(save_address);
filtdata=filtdata';
filtdata=filtdata(:);
figure(1);
subplot(2,1,1);
plot(data);
subplot(2,1,2);
plot(filtdata);

