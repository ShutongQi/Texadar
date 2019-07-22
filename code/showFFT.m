% clc;clear;
% name={'cjc','qst','smj'};
% % i=3;
% N=256;
% dataset=zeros(10,30,N);
% for i=1:3
% %     subplot(3,1,i);
%     for j=1:10
%         for k=1:10
%             address=['D:\2019summer\data\output\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.txt'];
%             data=importdata(address);
%             data=data';
%             data=data(:);
%             offset=mean(data(1:128));
%             data=data-offset;
%             Fdata=fftshift(fft(data,N));
%             amp=abs(Fdata);
%             dataset(j,10*(i-1)+k,:)=amp;
% %             plot(abs(Fdata));
% %             hold on
%         end
%     end
% end
%% Normalize
% m=zeros(10,1);
% normdata=zeros(10,30,256);
% for i=1:10
%     m(i)=max(max(dataset(i,:,:)));
%     normdata(i,:,:)=dataset(i,:,:)./m(i);
% end

%% plot normalize
% ges1=zeros(30,256);
% for i=1:30
%     ges1(i,:)=normdata(2,i,:);
%     plot(ges1(i,:));
%     hold on
% end

%% 
% data=zeros(256,1);
% for i=1:10
%     for j=1:30
%         
%         data=normdata(i,j,:);
%         data=reshape(data,1,[]);
%         num=(i-1)*30+j;
%         address=['D:\2019summer\data\dataset1\',num2str(num),'.mat'];
%         save(address,'data');
%     end
% end

%% 
for i=1:10
    address=['D:\2019summer\data\dataset1\',num2str(i),'.mat'];
    data=load(address);
    plot(data.data)
    hold on
end