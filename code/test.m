clc;clear;
%% 
name={'cjc','qst','smj'};
for i=1:3
    figure(1);
    address=['D:\2019summer\data\dataset\',num2str(270+10*i),'.mat'];
    data=load(address);
    color={'r','g','b'};
    plot(data.data,color{i});
    hold on
end

for i=1:2
    figure(2);
    address=['D:\2019summer\data\dataset\',num2str(30*i),'.mat'];
    data=load(address);
    plot(data.data);
    hold on
end
% for i=1:10
%     figure(3)
%     address=['D:\2019summer\data\output\',name{1},'_',num2str(1),'_',num2str(i-1),'_filtered.txt'];
%     orgdata=importdata(address);
%     orgdata=orgdata';
%     orgdata=reshape(orgdata,1,[]);
%     plot(orgdata);
%     hold on
% end
% for j=1:10
%     figure(4)
%     subplot(2,5,j);
%     for i=1:3
%         address=['D:\2019summer\data\output\',name{i},'_',num2str(j),'_',num2str(1),'_filtered.txt'];
%         orgdata=importdata(address);
%         orgdata=orgdata';
%         orgdata=reshape(orgdata,1,[]);
%         color={'r','g','b'};
%         plot(orgdata,color{i});
%         hold on
%     end
% end
%% 
% i=1;
% j=1;
% k=1;
% name={'cjc','qst','smj'};
% address=['D:\2019summer\data\gesture_filt\',name{i},'_',num2str(j),'_',num2str(k),'.txt'];
% orgdata=importdata(address);
% orgdata=orgdata';
% orgdata=reshape(orgdata,1,[]);
% data=orgdata(1:25000);
% % plot(data);
% Fdata=abs(fftshift(fft(data,64)));
% Fdata(30:36)=(Fdata(32)+Fdata(34))/2;
% plot(Fdata);

%% 
% name={'cjc','qst','smj'};
% i=1;
% j=2;
% k=2;
% % allFdata=zeros(300,1024);
% address=['D:\2019summer\data\output\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.txt'];
% %             save_address=['D:\2019summer\data\dataset\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.mat'];
% orgdata=importdata(address);
% orgdata=orgdata';
% orgdata=reshape(orgdata,1,[]);
% %             plot(orgdata);
% orgdata=reshape(orgdata,[],16);
% orgdata=orgdata';
% Fdata=zeros(16,64);
% for t=1:16
%     Fdata(t,:)=fftshift(fft(orgdata(t,:),64));
%     Fdata(t,:)=abs(Fdata(t,:));
% %     Fdata(t,33)=0;
%     Fdata(t,27:39)=(Fdata(t,27)+Fdata(t,39))/2;
% %                     plot(abs(Fdata(t,:)));
%     %                 amp=abs(Fdata(t,:));
%     %                 [~,index]=max(amp);
%     %                 amp(index)=amp(index+1);
%     %                 Fdata(t,:)=amp;
%     %                 M=max(amp);
%     %                 Fdata(t,:)=Fdata(t,:)./M;
% end
% Fdata=abs(Fdata');
% Fdata=reshape(Fdata,1,[]);
% plot(Fdata);

%% 
% i=1;
% j=1;
% k=1;
% address=['D:\2019summer\data\gesture_normNfilt\',name{i},'_',num2str(j),'_',num2str(k),'.txt'];
% orgdata=importdata(address);
% orgdata=orgdata';
% orgdata=reshape(orgdata,1,[]);
% figure(2);
% plot(orgdata);
