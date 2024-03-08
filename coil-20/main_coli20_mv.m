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

a1 = 10;
X = reshape(X,N2,N1,[]);
X(2:2:end,a1+1:end,:) = 0;
X(2:2:end,a1+1:end,:) = X(2:2:end,a1+1:end,:) + mean(X(2:2:end,1:a1,:),2) + mean(X(1:2:end,a1+1:end,:),1);
X = reshape(X,N1*N2,[]);
mask = false(N2,N1);
mask(2:2:end,a1+1:end) = true;
mask = reshape(mask,N2*N1,1);
mean_impute = 0;

%% main mwgl
if mean_impute == 1
tic;
alpha_mwgl = [0.05,0.02,0.01,0.005,0.002,0.001];
len_alpha_mwgl = length(alpha_mwgl);
    
graphs1_mwgl = zeros(N1*(N1-1)/2,len_alpha_mwgl,len_alpha_mwgl);
graphs2_mwgl = zeros(N2*(N2-1)/2,len_alpha_mwgl,len_alpha_mwgl);
for i = 1:len_alpha_mwgl
    for j = 1:len_alpha_mwgl
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha_mwgl(i)*N2,alpha_mwgl(j)*N1];
        param.pd_type = 'cartesian';%'tensor';%'strong';%
        param.inv_compute = 'eig';
        param.max_iter = 10000; % 10000 get double circle
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = mwgl(X,param);
        w1 = -L1(tril(true(N1),-1));
        w2 = -L2(tril(true(N2),-1));
        graphs1_mwgl(:,i,j) = w1;
        graphs2_mwgl(:,i,j) = w2;
    end
end
toc;
end

%% main mwgl_mv
tic;
alpha_mwgl_mv = [0.03,0.02,0.01,0.005];
len_alpha_mwgl_mv = length(alpha_mwgl_mv);
    
graphs1_mwgl_mv = zeros(N1*(N1-1)/2,len_alpha_mwgl_mv);
graphs2_mwgl_mv = zeros(N2*(N2-1)/2,len_alpha_mwgl_mv);
impute = zeros(N1*N2,M,len_alpha1);
% parfor (i = 1:len_alpha_mwgl_mv, 4)
for i = 1:len_alpha_mwgl_mv
    param = struct();
    param.N1 = N1;
    param.N2 = N2;
    param.alpha = [alpha_mwgl_mv(i)*N2,alpha_mwgl_mv(i)*N1];
    param.pd_type = 'cartesian';%'tensor';%'strong';%
    param.inv_compute = 'eig';
    param.max_iter = 20000; % 10000 get double circle
    param.step_size = 1e-3;
    param.tol = 1e-6;
    param.mask = mask;
    param.beta = 1;
    [L,L1,L2,X_impute] = mwgl_mv(X,param);
    impute(:,:,i) = X_impute;
    w1 = -L1(tril(true(N1),-1));
    w2 = -L2(tril(true(N2),-1));
    graphs1_mwgl_mv(:,i) = w1;
    graphs2_mwgl_mv(:,i) = w2;
end
toc;

%% viz
figure(1);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G1 = graph(squareform(graphs1_mwgl_mv(:,4)));
LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
c = summer;
h = plot(G1,'LineWidth',LWidths1,'NodeFontSize',5,'MarkerSize',8,'Marker','o','NodeColor',c(1,1:3),'EdgeColor',0.6*[1,1,1],'EdgeAlpha',1);
layout(h,'force','Iterations',50,'UseGravity','on')
rotate(h,[0,0,1],60);hold on;
ax = gca;
set(gca,'Visible','off')
% exportgraphics(ax,'object_graph_ours_alpha1=04_alpha2=03_impute1_rotate.png','BackgroundColor','none',Resolution=600)
%%
figure(2);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G2 = graph(squareform(graphs2_mwgl(:,3,4)));
LWidths2 = 5*G2.Edges.Weight/max(G2.Edges.Weight);
c = hsv(18);
cc = vertcat(c,c);
z = colormap(cc*0.7);
plot(G2,'LineWidth',LWidths2,'MarkerSize',8,'Marker','o','NodeLabel',{},'NodeColor',z,'EdgeColor',0.6*[1,1,1],'EdgeAlpha',1);
alpha(0.1);
set(gca,'Visible','off');
ax = gca;
cbar = colorbar;
cbar.Ticks = linspace(0,1,5);
cbar.TickLabels = ["0^{\circ}","90^{\circ}","180^{\circ}","270^{\circ}","360^{\circ}"];%0:90:360;
cbar.AxisLocation = "out";
cbar.FontSize = 13;
pos = cbar.Position;
cbar.Position = [pos(1)+0.2*pos(3),pos(2)+0.1*pos(4),pos(3),pos(4)*0.8];
% exportgraphics(ax,'angle_graph_ours_alpha1=04_alpha2=03_hsv07.jpg','BackgroundColor','none',Resolution=600)
%%
figure(3);
tiledlayout(5,10, 'Padding', 'none', 'TileSpacing', 'compact'); 
selected_impute = impute(:,:,4);
selected_impute(~mask,:) = X(~mask,:);

for i = 1:5
    for j = 1:10
        nexttile(10*(i-1)+j)
        imagesc(reshape(selected_impute(N2*(i+9+5)+j*2,:),128,128));
        xticks([]);yticks([]);
        colormap gray;
        axis square;
        if i == 5
            xlabel(strcat([num2str(-20+20*j),'\circ']),'FontSize',10);
        end
    end
end
% print('imputation_ours_alpha1=02_alpha2=02_label','-djpeg','-r600');