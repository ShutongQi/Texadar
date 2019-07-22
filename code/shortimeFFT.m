clc;clear;
name={'cjc','qst','smj'};
% i=3;
allFdata=zeros(300,1024);
for i=1:3
    for j=1:10
        for k=1:10
            address=['D:\2019summer\data\output\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.txt'];
            %             save_address=['D:\2019summer\data\dataset\',name{i},'_',num2str(j),'_',num2str(k-1),'_filtered.mat'];
            orgdata=importdata(address);
            orgdata=orgdata';
            orgdata=reshape(orgdata,1,[]);
%             plot(orgdata);
%             temp=orgdata;
            orgdata=reshape(orgdata,[],16);
            orgdata=orgdata';
            Fdata=zeros(16,64);
            temp1=zeros(1,64);
            temp2=zeros(1,64);
            for t=1:16
                Fdata(t,:)=fftshift(fft(orgdata(t,:),64));
                Fdata(t,:)=abs(Fdata(t,:));
                Fdata(t,31:35)=(Fdata(t,31)+Fdata(t,35))/2;
                temp2=Fdata(t,:);
                Fdata(t,:)=Fdata(t,:)-temp1;
                temp1=temp2;
                
%                 plot(abs(Fdata(t,:)));
%                 amp=abs(Fdata(t,:));
%                 [~,index]=max(amp);
%                 amp(index)=amp(index+1);
%                 Fdata(t,:)=amp;
%                 M=max(amp);
%                 Fdata(t,:)=Fdata(t,:)./M;
            end
            Fdata=abs(Fdata');
            Fdata=reshape(Fdata,1,[]);
            
            num=(j-1)*30+(i-1)*10+k;
            allFdata(num,:)=Fdata;
        end
    end
end
%% 

M=max(max(allFdata));
normFdata=allFdata./M;

%% 
for i=1:300
    data=normFdata(i,:);
    save_address=['D:\2019summer\data\dataset\',num2str(i),'.mat'];
    save(save_address,'data');
end

