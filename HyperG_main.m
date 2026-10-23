%%  SAR Image Change Detection based on High-order Structure Consistency between Hypergraphs
clc;
close all;
clear;

%% 1- Data Reading
addpath(genpath(pwd));

dataName = 'Y4';   
X = imread(dataName+ "" + '1.png');
Y = imread(dataName+ "" + '2.png');
Ref_gt = imread(dataName+ "" + '3.png');
ref=Ref_gt(:,:,1);

X_nor = image_normlized(X,'sar');  % SAR or optical image type
Y_nor = image_normlized(Y,'sar');
clear X Y;

tStart = tic;  % overall timing start

%% Parameter setting and patch segmentation
% With different parameter settings, the results will be a little different
[m,n] = size(X_nor(:,:,1)); 
adj_rate = 0.01; % adjacency ratio
ps = 2;   % patch size
ss = 2;    % patch segmentation stride

X_p = imageTodata(X_nor,ps,ss);   % convert the image into a data matrix with each patch as a column
Y_p = imageTodata(Y_nor,ps,ss);   % patch-based conversion
MM = max(size(X_p));
K =round(MM*adj_rate);
k = K+1;   % self-loop
X_p_t = X_p';
Y_p_t = Y_p';

T1 = toc(tStart); fprintf('Patch segmentation：%.4f s\n', T1);
%%  Construct the hypergraph based on patches
[idx, distX] = knnsearch(X_p_t,X_p_t,'k',k); 
[idy, distY] = knnsearch(Y_p_t,Y_p_t,'k',k);  
T2 = toc(tStart); fprintf('Hypergraph Construct：%.4f s\n', T2-T1);

%% Hyperedge weight and hyperedge similarity
[DI_e, DI_he] = Highorder_Structure_Consistency(X_p_t, Y_p_t, idx, idy, k, ps, ss, X_nor);
T3 = toc(tStart); fprintf('Consistency Measure：%.4f s\n', T3-T2);

%% Difference image generation
DI =  (1+2*DI_e).*DI_he;
DI = DI./max(DI(:));
% figure, imshow(DI);

indexedImage = gray2ind(uint8(DI*255), 256); 
rgbImage = ind2rgb(indexedImage, jet(256)); 
figure, imshow(rgbImage);

T4 = toc(tStart); fprintf('Difference image generation：%.4f s\n', T4-T3);

%% Otsu threshold
level=graythresh(DI)
CM_OTSU = im2bw(DI, level);

TimeCT = T4;
%% Result visualization
ref = ref/max(ref(:));

[PRE, REC] = PR_plot(DI,ref,500);
[TPR, FPR]= Roc_plot(DI,ref, 500);
[AUC,~] = AUC_Diagdistance(TPR, FPR);
[AUP, ~] = AUP_Diagdistance(PRE, REC);

ROC_HyperG = [FPR; TPR];
PRC_HyperG = [REC, PRE];
% figure; plot(FPR,TPR);title('ROC curves'); 
% figure; plot(REC,PRE);title('PR curves');

[tp,fp,tn,fn,fplv,fnlv,~,~,pcc,kappa,imw]=performance(CM_OTSU,1*ref);
F1 = 2*tp/(2*tp + fp + fn);

[FP_x, FP_y] = find(imw==0);
[FN_x, FN_y] = find(imw==255);
CM_map_OTSU(:,:,1) = uint8(CM_OTSU)*255;
CM_map_OTSU(:,:,2) = uint8(CM_OTSU)*255;
CM_map_OTSU(:,:,3) = uint8(CM_OTSU)*255;
for i = 1 : max(size(FP_x))
    CM_map_OTSU(FP_x(i), FP_y(i), 1:3) = [255 0 0];
end
for i = 1 : max(size(FN_x))
    CM_map_OTSU(FN_x(i), FN_y(i), 1:3) = [0 255 0];
end

filename_OTSU = sprintf(dataName + "" + '_OTSU_HyperG_FN is %d; FP is %d; AUC is %4.4f; AUP is %4.4f;  PCC is %4.4f; F1 is %4.4f; KC is %4.4f.png',   fn, fp, AUC, AUP, pcc, F1, kappa)

%% MTEP

ii=1;
for i_oa = 0 : 0.001 : 1
    CM_METP = im2bw(DI, i_oa); 
    [~,~,~,~,~,~,~,~,OA_id(ii),~,~]  = performance(CM_METP,1*ref);
    ii=ii+1;
end
id_oa = min(find(OA_id==max(OA_id))*0.001);

CM_METP_f = im2bw(DI, id_oa); 
[tp,fp,tn,fn,fplv,fnlv,~,~,pcc,kappa,imw] = performance(CM_METP_f,1*ref);
F1 = 2*tp/(2*tp + fp + fn);

[FP_x, FP_y] = find(imw==0);
[FN_x, FN_y] = find(imw==255);
CM_map_METP(:,:,1) = uint8(CM_METP_f)*255;
CM_map_METP(:,:,2) = uint8(CM_METP_f)*255;
CM_map_METP(:,:,3) = uint8(CM_METP_f)*255;
for i = 1 : max(size(FP_x))
    CM_map_METP(FP_x(i), FP_y(i), 1:3) = [255 0 0];
end
for i = 1 : max(size(FN_x))
    CM_map_METP(FN_x(i), FN_y(i), 1:3) = [0 255 0];
end
figure,imshow(CM_map_METP);
filename_METP = sprintf(dataName + "" + '_METP_HyperG_%4.2fs_FN is %d; FP is %d; AUC is %4.4f; AUP is %4.4f; PCC is %4.4f; F1 is %4.4f; KC is %4.4f.png',  TimeCT, fn, fp, AUC, AUP,  pcc, F1, kappa)

% save(dataName + "" + '_ROC_HyperG.mat', 'ROC_HyperG');
% save(dataName + "" + '_PRC_HyperG.mat', 'PRC_HyperG');
% filename_DI = sprintf(dataName + "" + '_DI_HyperG.png'); 
% filename_PC = sprintf(dataName + "" + '_PC_HyperG.png'); 
% % imwrite(DI, filename_DI);
% imwrite(rgbImage, filename_PC);
% imwrite(CM_map_OTSU, filename_OTSU);
% imwrite(CM_map_METP, filename_METP);