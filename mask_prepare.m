addpath('training/training');
label=importdata('label.xlsx');
label=round(label.data.Sheet1);
k=1;count=1;
for i = 60091:70090
    fprintf('loading sample...%d\n',i)
    fname=sprintf('%d',i);
    image=importdata([fname '.jpg']);
    image_chan1=image(:,:,1);%only 1 channel is enough
    label_temp=label(k,:);
    index=find(label_temp==1);
    for j = 1:numel(index)
        x=label_temp(:,index(j)-4);y=label_temp(:,index(j)-3);
        length=label_temp(:,index(j)-2)-x;width=label_temp(:,index(j)-1)-y;
        mask_temp=zeros(size(image_chan1));
        mask_temp(y:(y+width),x:(x+length))=1;
        mask_sub(j,:)=mask_temp(:);
    end
    if j>1
        mask_sub_tot=sum(mask_sub);
    else
        mask_sub_tot=mask_sub;
    end
    mask(k,:)=mask_sub_tot(:);
end
% label2_temp=round(label2);