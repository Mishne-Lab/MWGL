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
X0 = X;
X = X/255-0.5;

%% main loop

%% main rpgl
tic;
beta1_rpgl = 0.1.^(-1:0.2:1); % 6-7
beta2_rpgl = 0.1.^(0:0.2:2); % 6-7
len_beta1_rpgl = length(beta1_rpgl);
len_beta2_rpgl = length(beta2_rpgl);
    
for i = 1:len_beta1_rpgl
    for j = 1:len_beta2_rpgl
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.solver = 'idqp';
        param.beta1 = beta1_rpgl(i);
        param.beta2 = beta2_rpgl(j);
        param.rho = 0.001;
        param.tol = 1e-6;
        param.max_iter = 20000;
        param.thr = 0;
        [L,L1,L2] = rpgl(X,param);
        graphs1_rpgl(:,i,j) = -L1(tril(true(N1),-1));
        graphs2_rpgl(:,i,j) = -L2(tril(true(N2),-1));
    end
end
toc;

%% main teralasso
tic;
X_rs = reshape(X,N2,N1,[]);
T = reshape(X_rs,N2,[])*reshape(X_rs,N2,[])'/N1/M;
X_rs = permute(X_rs,[2,1,3]);
S = reshape(X_rs,N1,[])*reshape(X_rs,N1,[])'/N2/M;
lambda1_teralasso = 0.1.^[1:0.2:2];%0;%
lambda2_teralasso = 0.1.^[0.8:0.1:1.2];
% lambda = [lambda,0];
len_lambda1_teralasso = length(lambda1_teralasso);
len_lambda2_teralasso = length(lambda2_teralasso);
tol = 1e-4;
maxiter = 2000;
for i = 1:len_lambda1_teralasso
    for j = 1:len_lambda2_teralasso
        [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda1_teralasso(i),lambda2_teralasso(j)],maxiter);
        L1 = PsiH{1};
        L2 = PsiH{2};
        graphs1_teralasso(:,i,j) = L1(:);
        graphs2_teralasso(:,i,j) = L2(:);
    end
end
toc;

%% main mwgl
tic;
alpha_mwgl = [0.075,0.05,0.04,0.03,0.02,0.01,0];
len_alpha_mwgl = length(alpha_mwgl);
    
graphs1_mwgl = zeros(N1*(N1-1)/2,len_alpha_mwgl,len_alpha_mwgl);
graphs2_mwgl = zeros(N2*(N2-1)/2,len_alpha_mwgl,len_alpha_mwgl);
% parfor (i = 1:len_alpha1, 5)
for i = 1:len_alpha_mwgl
    for j = 1:len_alpha_mwgl
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha_mwgl(i)*N2,alpha_mwgl(j)*N1];
        param.pd_type = 'cartesian';
        param.inv_compute = 'eig';
        param.max_iter = 10000; % 10000 get double circle
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = blpgl(X,param);
        w1 = -L1(tril(true(N1),-1));
        w2 = -L2(tril(true(N2),-1));
        graphs1_mwgl(:,i,j) = w1;
        graphs2_mwgl(:,i,j) = w2;
    end
    
end
toc;

%% viz
figure(1);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G1 = graph(squareform(graphs1_blpgl(:,3,4)));
LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
c = summer;
h = plot(G1,'LineWidth',LWidths1,'NodeFontSize',15,'MarkerSize',8,'Marker','o','NodeColor',c(1,1:3),'EdgeColor',0.6*[1,1,1],'EdgeAlpha',1,"NodeLabel",{});
ax = gca;
set(gca,'Visible','off')
% Add new labels that are to the upper, right of the nodes
text(h.XData-.01, h.YData+.02 ,h.NodeLabel, ...
    'VerticalAlignment','Bottom',...
    'HorizontalAlignment', 'left',...
    'FontSize', 15)
% Remove old labels
h.NodeLabel = {}; 
% exportgraphics(ax,'object_graph_ours_alpha1=04_alpha2=03_nolabel.jpg','BackgroundColor','none',Resolution=600)
%%
figure(12);
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
nexttile
G2 = graph(squareform(graphs2_mwgl(:,3,4)));
LWidths2 = 5*G2.Edges.Weight/max(G2.Edges.Weight);
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
% exportgraphics(ax,'angle_graph_ours_alpha1=04_alpha2=03_hsv07.jpg','BackgroundColor','none',Resolution=600)

%%
figure(2);
G1 = graph(squareform(graphs1_blpgl(:,3,4)));
edges = G1.Edges.EndNodes;
LWidths1 = 5*G1.Edges.Weight/max(G1.Edges.Weight);
% LWidths1 = 5*log(1+G1.Edges.Weight)/max(log(1+G1.Edges.Weight));
plot(G1,'LineWidth',LWidths1,'NodeFontSize',15,'MarkerSize',8);%,'NodeColor',[0.6,0.6,0.6]
ax = gca;
set(gca,'Visible','off')
% exportgraphics(ax,'obj_graph_ours_alpha1=04_alpha2=03_blue.jpg',Resolution=600)
[sorted_weights,idx] = sort(LWidths1,'descend');

figure(3);
tcl = tiledlayout(2,5);
tcl.TileSpacing = 'tight';
tcl.Padding = 'tight';
% sgtitle('Relative Frobenius Error of Laplacian');
for i = 1:5
    nexttile(i);
    i1 = edges(idx(i),1);
    i2 = edges(idx(i),2);
    x1 = X0((i1-1)*N2+1,:);
    imagesc(reshape(x1,128,128));axis square;axis off;
    xticks([]);
    colormap gray;
    title(strcat([num2str(i1),'-',num2str(i2)]),'FontWeight','normal','FontSize',25);
    nexttile(i+5);
    x2 = X0((i2-1)*N2+1,:);
    imagesc(reshape(x2,128,128));axis square;axis off;
    colormap gray;
end