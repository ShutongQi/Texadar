% trainset=fopen('trainset.txt','w');
% for i=1:10
%     for j=1:25
%     label=i-1;
%     str=['D:\2019summer\data\dataset\',num2str((i-1)*30+j),'.mat ',num2str(label)];
%     fprintf(trainset,'%s\r\n',str);
%     end
% end
%% 
% trainset=fopen('testset.txt','w');
% for i=1:10
%     for j=1:5    
%     label=i-1;
%     num=(i-1)*30+j;
%     str=['D:\2019summer\data\dataset\',num2str(num+25),'.mat ',num2str(label)];
%     fprintf(trainset,'%s\r\n',str);
%     end
% end

%% 
trainset=fopen('dataset.txt','w');
for i=1:300
    
    label=ceil(i/30)-1;
    str=['D:\2019summer\data\dataset\',num2str(i),'.mat ',num2str(label)];
    fprintf(trainset,'%s\r\n',str);
    
end