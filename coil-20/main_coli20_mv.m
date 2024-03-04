clear;close all;
%% read images
N1 = 20;
N2 = 72;
M = 128*128;

% subsample angles
angle = 2;
N2 = N2/angle;
X = zeros(N1*N2,M);

for i = 1:N1
    for j = 1:N2
        img = imread(join(["coil-20-proc/obj",num2str(i),"__",num2str((j-1)*angle),".png"],""));
        X((i-1)*N2+j,:) = img(:);
    end
end
X = X/255-0.5;

%% missing values
% a1 = 10; %15
% a2 = 18; %24
% X = reshape(X,N2,N1,[]);
% X(a2+1:end,a1+1:end,:) = 0;
% X(a2+1:end,a1+1:end,:) = X(a2+1:end,a1+1:end,:) + mean(X(a2+1:end,1:a1,:),2) + mean(X(1:a2,a1+1:end,:),1);
% X = reshape(X,N1*N2,[]);
% mask = false(N2,N1);
% mask(a2+1:end,a1+1:end) = true;
% mask = reshape(mask,N2*N1,1);

a1 = 10;
X = reshape(X,N2,N1,[]);
X(2:2:end,a1+1:end,:) = 0;
X(2:2:end,a1+1:end,:) = X(2:2:end,a1+1:end,:) + mean(X(2:2:end,1:a1,:),2) + mean(X(1:2:end,a1+1:end,:),1);
X = reshape(X,N1*N2,[]);
mask = false(N2,N1);
mask(2:2:end,a1+1:end) = true;
mask = reshape(mask,N2*N1,1);
% D = pdist2(X,X,'squaredeuclidean');
% for i = 1:N1
%     for j = 1:N2
%         if i>15 && j>24
%             D((i-1)*N2+j,:) = 0;
%             D(:,(i-1)*N2+j) = 0;
%         end
%     end
% end
% D_rs = reshape(D,N2,N1,N2,N1);
% D_rs = permute(D_rs,[1,3,2,4]);
% D2 = zeros(N2);
% for j = 1:N1
%     D2 = D2+D_rs(:,:,j,j);
% end
% D_rs = permute(D_rs,[3,4,1,2]);
% D1 = zeros(N1);
% for i = 1:N2
%     D1 = D1+D_rs(:,:,i,i);
% end
mean_impute = 0;

%% main blpgl
if mean_impute == 1
tic;
alpha1 = [0.05,0.02,0.01,0.005,0.002,0.001];
alpha2 = [0.05,0.02,0.01,0.005,0.002,0.001];
    len_alpha1 = length(alpha1);
    len_alpha2 = length(alpha2);
    
graphs1_blpgl = zeros(N1*(N1-1)/2,len_alpha1,len_alpha2);
graphs2_blpgl = zeros(N2*(N2-1)/2,len_alpha1,len_alpha2);
for i = 1:len_alpha1
    for j = 1:len_alpha2
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha1(i)*N2,alpha2(j)*N1];
        param.pd_type = 'cartesian';%'tensor';%'strong';%
        param.inv_compute = 'eig';
        param.max_iter = 10000; % 10000 get double circle
        param.step_size = 1e-3;
        param.tol = 1e-6;
%         [L,L1,L2] = blpgl2(X,D1,D2,param);
        [L,L1,L2] = blpgl(X,param);
        w1 = -L1(tril(true(N1),-1));
        w2 = -L2(tril(true(N2),-1));
        graphs1_blpgl(:,i,j) = w1;
        graphs2_blpgl(:,i,j) = w2;
    end
    
end
toc;
end

%% main blpgl_mv
tic;
alpha1 = [0.05,0.04,0.03,0.02,0.01,0.005,0.001,0];
alpha2 = [0.05,0.04,0.03,0.02,0.01,0.005,0.001,0];
alpha1 = [0.03,0.02,0.01,0.005];
alpha2 = [0.03,0.02,0.01,0.005];
    len_alpha1 = length(alpha1);
    len_alpha2 = length(alpha2);
    
graphs1_blpgl_mv = zeros(N1*(N1-1)/2,len_alpha1);
graphs2_blpgl_mv = zeros(N2*(N2-1)/2,len_alpha1);
% parfor (i = 1:len_alpha1, 6)
impute = zeros(N1*N2,M,len_alpha1);
% parfor (i = 1:len_alpha1, 6)
for i = 1:len_alpha1
%     for j = 1:len_alpha2
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha1(i)*N2,alpha2(i)*N1];
        param.pd_type = 'cartesian';%'tensor';%'strong';%
        param.inv_compute = 'eig';
        param.max_iter = 20000; % 10000 get double circle
        param.step_size = 1e-3;
        param.tol = 1e-6;
        param.mask = mask;
        param.beta = 1;
%         [L,L1,L2] = blpgl2(X,D1,D2,param);
        [L,L1,L2,X_impute] = blpgl_mv(X,param);
        impute(:,:,i) = X_impute;
        w1 = -L1(tril(true(N1),-1));
        w2 = -L2(tril(true(N2),-1));
        graphs1_blpgl_mv(:,i) = w1;
        graphs2_blpgl_mv(:,i) = w2;
%     end
    
end
toc;

%% viz
figure(5);
for i = 1:len_alpha1
    subplot(3,3,i)
%     imagesc(squareform(graphs1_blpgl_mv(:,i)));
        G1 = graph(squareform(graphs1_blpgl_mv(:,i)));
        LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
        H1 = plot(G1,'LineWidth',LWidths1);
%         layout(H1,'circle');
        layout(H1,'force','Iterations',100,'UseGravity','on');
end

figure(6);
for i = 1:len_alpha1
    subplot(3,3,i)
%     imagesc(squareform(graphs2_blpgl_mv(:,i)));
    G2 = graph(squareform(graphs2_blpgl_mv(:,i)));
    LWidths2 = 5*G2.Edges.Weight/max(G2.Edges.Weight);
    H2 = plot(G2,'LineWidth',LWidths2);
%     layout(H2,'force','Iterations',500);
end

%% viz
figure(5);
for i = 1:len_alpha1
    for j = 1:len_alpha2
        subplot(len_alpha1,len_alpha2,(i-1)*len_alpha2+j)
        imagesc(squareform(graphs1_blpgl(:,i,j)));
%         G1 = graph(squareform(graphs1_blpgl(:,i,j)));
%         LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
%         plot(G1,'LineWidth',LWidths1);
    end
end

figure(6);
for i = 1:len_alpha1
    for j = 1:len_alpha2
        subplot(len_alpha1,len_alpha2,(i-1)*len_alpha2+j)
    %     imagesc(squareform(graphs2_blpgl(:,i)));
        G2 = graph(squareform(graphs2_blpgl(:,i,j)));
        LWidths2 = 5*G2.Edges.Weight/max(G2.Edges.Weight);
        plot(G2,'LineWidth',LWidths2);
    end
end

%%
figure(7);
for i = 1:N1
    img = imread(join(["coil-20-proc/obj",num2str(i),"__0.png"],""));
    subplot(4,5,i);
%     subplot(1,20,i);
    imagesc(img);
    axis off;
    colormap gray;
    clim([0,255]);
end
sgtitle("object catelog")

%%
figure(11);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G1 = graph(squareform(graphs1_blpgl_mv(:,4)));
LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
% LWidths1 = 5*log(1+G1.Edges.Weight)/max(log(1+G1.Edges.Weight));1-(1-[0 0.4470 0.7410])*0.75
c = summer;
h = plot(G1,'LineWidth',LWidths1,'NodeFontSize',5,'MarkerSize',8,'Marker','o','NodeColor',c(1,1:3),'EdgeColor',0.6*[1,1,1],'EdgeAlpha',1);
layout(h,'force','Iterations',50,'UseGravity','on')
rotate(h,[0,0,1],60);hold on;
ax = gca;
set(gca,'Visible','off')
% % Add new labels that are to the upper, right of the nodes
% text(h.XData-.01, h.YData+.02 ,h.NodeLabel, ...
%     'VerticalAlignment','Bottom',...
%     'HorizontalAlignment', 'left',...
%     'FontSize', 15)
% % Remove old labels
% h.NodeLabel = {}; 
exportgraphics(ax,'object_graph_ours_alpha1=04_alpha2=03_impute1_rotate.png','BackgroundColor','none',Resolution=600)
%%
figure(12);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G2 = graph(squareform(graphs2_blpgl(:,3,4)));
LWidths2 = 5*G2.Edges.Weight/max(G2.Edges.Weight);
% z = redblue(36,[-18,18]);
% cool_wrap = vertcat(summer(18),flipud(summer(18)));
% z = colormap(cool_wrap*0.8);
% c = [0.9612 0.4459 0.4459;0.9475 0.4615 0.4019;0.9305 0.4783 0.3598;0.9102 0.4963 0.3201;0.887 0.5154 0.283;0.8611 0.5352 0.2489;0.8326 0.5557 0.2183;0.8019 0.5765 0.1913;0.7692 0.5977 0.1683;0.7349 0.6188 0.1494;0.6993 0.6397 0.1348;0.6628 0.6603 0.1248;0.6256 0.6803 0.1193;0.5882 0.6995 0.1184;0.5509 0.7178 0.1222;0.5141 0.7349 0.1306;0.4781 0.7507 0.1435;0.4433 0.7651 0.1608;0.41 0.7779 0.1823;0.3785 0.789 0.2079;0.3492 0.7982 0.2372;0.3222 0.8056 0.27;0.298 0.8109 0.306;0.2766 0.8143 0.3449;0.2584 0.8155 0.3861;0.2435 0.8147 0.4295;0.232 0.8118 0.4745;0.224 0.8069 0.5207;0.2197 0.8 0.5676;0.219 0.7912 0.6149;0.222 0.7805 0.662;0.2286 0.7681 0.7086;0.2388 0.7541 0.7541;0.2525 0.7385 0.7981;0.2695 0.7217 0.8402;0.2898 0.7037 0.8799;0.313 0.6846 0.917;0.3389 0.6648 0.9511;0.3674 0.6443 0.9817;0.4024 0.623 1;0.4432 0.6022 1;0.4802 0.5833 1;0.5146 0.5658 1;0.5472 0.5492 1;0.5787 0.5332 1;0.6098 0.5173 1;0.6411 0.5014 1;0.6732 0.485 1;0.7068 0.4679 1;0.7427 0.4496 1;0.782 0.4296 1;0.8215 0.411 0.9921;0.8508 0.4018 0.9628;0.8778 0.3944 0.93;0.902 0.3891 0.894;0.9234 0.3857 0.8551;0.9416 0.3845 0.8139;0.9565 0.3853 0.7705;0.968 0.3882 0.7255;0.976 0.3931 0.6793;0.9803 0.4 0.6324;0.981 0.4088 0.5851;0.978 0.4195 0.538;0.9714 0.4319 0.4914];
c = hsv(18);
cc = vertcat(c,c);
z = colormap(cc*0.7);
% z = colormap(pmkmp(36,'IsoAZ180'));
plot(G2,'LineWidth',LWidths2,'MarkerSize',8,'Marker','o','NodeLabel',{},'NodeColor',z,'EdgeColor',0.6*[1,1,1],'EdgeAlpha',1);
alpha(0.1);
% axis square;
set(gca,'Visible','off');
ax = gca;
cbar = colorbar;
cbar.Ticks = linspace(0,1,5);
cbar.TickLabels = ["0^{\circ}","90^{\circ}","180^{\circ}","270^{\circ}","360^{\circ}"];%0:90:360;
% cbar.Location = "south";
cbar.AxisLocation = "out";
cbar.FontSize = 13;
pos = cbar.Position;
% cbar.Position = [pos(1)+0.1*pos(3),pos(2),pos(3)*0.8,pos(4)];
cbar.Position = [pos(1)+0.2*pos(3),pos(2)+0.1*pos(4),pos(3),pos(4)*0.8];
exportgraphics(ax,'angle_graph_ours_alpha1=04_alpha2=03_hsv07.jpg','BackgroundColor','none',Resolution=600)
%%
figure(13);
tiledlayout(5,10, 'Padding', 'none', 'TileSpacing', 'compact'); 
selected_impute = impute(:,:,4);
selected_impute(~mask,:) = X(~mask,:);

for i = 1:5
    for j = 1:10
        nexttile(10*(i-1)+j)
%         imagesc(reshape(selected_impute(N2*(i+9+5)+mod(37-(j-1)*1,36),:),128,128));
        imagesc(reshape(selected_impute(N2*(i+9+5)+j*2,:),128,128));
        xticks([]);yticks([]);
        colormap gray;
        axis square;
%         axis off;
        if i == 5
            xlabel(strcat([num2str(-20+20*j),'\circ']),'FontSize',10);
        end
    end
end
% print('imputation_ours_alpha1=02_alpha2=02_label','-djpeg','-r600');