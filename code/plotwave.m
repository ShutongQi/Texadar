clc;clear;
figure(1);
subplot(3,1,1);
for i=1:10
    address1=['D:\2019summer\data\output\','smj_1_',num2str(i-1),'_filtered.txt'];
    orgdata=importdata(address1);
    orgdata=orgdata';
    data=orgdata(:);
%     data_filt=medfilt1(data,32);
    offset=mean(data(1:128));
    data=data-offset;
    plot(data);
%     axis([0 35000 -50 50]);
    hold on
end
subplot(3,1,2);
for i=1:10
    address1=['D:\2019summer\data\output\','qst_1_',num2str(i-1),'_filtered.txt'];
    orgdata=importdata(address1);
    orgdata=orgdata';
    data=orgdata(:);
%     data_filt=medfilt1(data,32);
    offset=mean(data(1:128));
    data=data-offset;
    plot(data);
%     axis([0 35000 -50 50]);
    hold on
end
subplot(3,1,3);
for i=1:10
    address1=['D:\2019summer\data\output\','smj_1_',num2str(i-1),'_filtered.txt'];
    orgdata=importdata(address1);
    orgdata=orgdata';
    data=orgdata(:);
    data_filt=medfilt1(data,32);
%     offset=mean(data_filt(1:2048));
%     data_filt=data_filt-offset;
    plot(data);
%     axis([0 35000 -50 50]);
    hold on
end
% L=length(data_filt);
% N=floor(L/2048);
% offset=mean(data_filt(1:2048));
% cnt=zeros(N,1);
% 
% for i=1:N
%     for j=(i-1)*2048+1:i*2048-1
%         if (data_filt(j)-offset)*(data_filt(j+1)-offset)<0
%             cnt(i)=cnt(i)+1;
%         end
%     end
%     text(1024*(2*i-1),350,num2str(cnt(i)));
% end
