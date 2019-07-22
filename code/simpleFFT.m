clc;clear;
name={'cjc','qst','smj'};
allFdata=zeros(300,512);
for i=1:3
    for j=1:10
        for k=1:10
            address=['D:\2019summer\data\output\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.txt'];
            orgdata=importdata(address);
            orgdata=orgdata';
            orgdata=reshape(orgdata,1,[]);
%             plot(orgdata);
            num=(j-1)*30+(i-1)*10+k;
            Fdata=abs(fft(orgdata,1024));
            filter=32;
%             Fdata(513-filter:513+filter)=(Fdata(513-filter)+Fdata(513+filter))/2;
            Fdata(1:filter)=0;
            Fdata=Fdata(1:512);
            allFdata(num,:)=Fdata;
        end
    end
end
%% 
% 
% M=max(max(allFdata));
% normFdata=allFdata./M;

%% 
for i=1:300
    data=allFdata(i,:);
    save_address=['D:\2019summer\data\dataset\',num2str(i),'.mat'];
    save(save_address,'data');
end