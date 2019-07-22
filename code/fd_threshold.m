address1=['D:\2019summer\data\gesture\','qst_10_',num2str(9),'.txt'];
orgdata=importdata(address1);
orgdata=orgdata';
data=orgdata(:);
data_filt=medfilt1(data,32);
offset=mean(data_filt(1:2048*2));
data_filt=data_filt-offset;
figure(1)
plot(data_filt);
axis([0 length(data_filt) -100 100]);
hold on

L=length(data_filt);
N=floor(L/2048);
energy=zeros(N,1);

for i=1:N
    plot([2048*i,2048*i],[-100 100],'r-');
    for j=(i-1)*2048+1:i*2048
        energy(i)=energy(i)+data_filt(j).^2;
    end
    energy(i)=round(energy(i)/2048);
    text(1024*(2*i-1)-128,90,num2str(energy(i)));
end