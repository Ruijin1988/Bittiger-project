clear
addpath('training/training');
label=importdata('label.xlsx');
label=ceil(label.data.Sheet1);
k=1;mask_x=36;mask_y=64;
for i = 60091:(60091+5000) %pick the first 5000 data
    fprintf('loading sample...%d\n',k)
    fname=sprintf('%d',i);
    image=importdata([fname '.jpg']);
    image_chan1=image(:,:,1);%only 1 channel is enough
    image_store(k,:)=image_chan1(:);
    label_temp=label(k,:);
    index=find(label_temp==1);%find label 1 items
    ind_delet=find(mod(index,5)~=0); %remove those mix label 1 from wrong index
    index(ind_delet) = []; 
    if numel(index)==0 % fill 0 if no object
        mask(k,:)=zeros(1,mask_x*mask_y);
        k=k+1;
        continue;
    end
    for j = 1:numel(index)
        x=label_temp(:,index(j)-4);y=label_temp(:,index(j)-3);
        if x==0
            x=x+1;
        end
        if y==0
            y=y+1;
        end
        length=label_temp(:,index(j)-2)-x;width=label_temp(:,index(j)-1)-y;
        mask_temp=zeros(size(image_chan1));
        mask_temp(y:(y+width),x:(x+length))=1;
        mask_temp(mask_temp>0)=1; %ignore overlapping vehicle for quick testing
        mask_temp=double(im2bw(imresize(mask_temp,[mask_x, mask_y]),0.0));
        mask_sub(j,:)=mask_temp(:);
    end
    if j>1
        mask_sub_tot=sum(mask_sub);
    else
        mask_sub_tot=mask_sub;
    end
    mask(k,:)=mask_sub_tot(:);
    k=k+1;
    clear mask_sub;clear mask_temp;clear mask_sub_tot;
end
% label2_temp=round(label2);