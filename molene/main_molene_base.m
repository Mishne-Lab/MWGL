clear;close all
%% Generate a graph
baselines.rpgl = 1;
baselines.blpgl = 1;
baselines.teralasso = 1;

load("meteo_molene_t.mat");
X = value;
x = info{4}; y = info{3}; z = info{5}; coords = [x,y,5*z];
N2 = size(X,1);
N1 = 24;
X = reshape(X,[],24,31);
X = X(:,:,1:30);
X = X-mean(X,[1,2]);
X = reshape(X(:,:,1:30),[],30);
X = X-mean(X(:));
X = X/std(X(:));
M = size(X,2);

X_M = X;

%% main rpgl loop
if baselines.rpgl == 1
    tic;
    
    beta1 = 0.1.^(0:0.2:2);
    beta2 = 0.1.^(0:0.2:2);
    beta1 = [beta1,0];
    beta2 = [beta2,0];
    len_beta1 = length(beta1);
    len_beta2 = length(beta2);
    
    for i = 1:len_beta1
        for j = 1:len_beta2
            param = struct();
            param.N1 = N1;
            param.N2 = N2;
            param.solver = 'idqp';
            param.beta1 = beta1(i);
            param.beta2 = beta2(j);
            param.rho = 0.001;
            param.tol = 1e-6;
            param.max_iter = 20000;
            param.thr = 0.1;
            [L,L1,L2] = rpgl(X_M,param);
            LL1 = L1;
            LL2 = L2;
            L(abs(L)<10^(-4))=0;
            L1(abs(L1)<0.1)=0;
            L2(abs(L2)<0.1)=0;
            graphs1_rpgl(:,i,j) = -LL1(tril(true(N1),-1));
            graphs2_rpgl(:,i,j) = -LL2(tril(true(N2),-1));
        end
    end
    
    toc;
end

%% main teralasso loop
if baselines.teralasso == 1
    tic;

    X_M_rs = reshape(X_M,N2,N1,[]);
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(X_M_rs,[2,1,3]);
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;
    lambda = 0.1.^[-0.5:0.1:0.5];%0;%
    lambda = [lambda,0];
    len_lambda = length(lambda);
    tol = 1e-7;
    maxiter = 10000;
    for i = 1:12
        for j = 1:12
            [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda(i),lambda(j)],maxiter);
            LL1 = PsiH{1};
            LL2 = PsiH{2};
            L1 = PsiH{1};
            L2 = PsiH{2};
            L1(L1>0) = 0;
            L2(L2>0) = 0;
            L1 = -diag(sum(L1,1))+L1;
            L2 = -diag(sum(L2,1))+L2;
            graphs1_teralasso2(:,i,j) = LL1(:);
            graphs2_teralasso2(:,i,j) = LL2(:);
        end
    end
    toc;
end

%% main mwgl loop
if baselines.blpgl == 1
    tic;
    
    alpha = 0.1.^(1:0.2:3);
    alpha = [alpha,0];%
    len_alpha = length(alpha);
    
    for i = 1:12
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha(i)*N2,alpha(i)*N1];
        param.pd_type = 'cartesian';
        param.inv_compute = 'eig';
        param.max_iter = 5000;
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = mwgl(X_M,param);
        LL1 = L1;
        LL2 = L2;
        graphs1_blpgl(:,i) = -LL1(tril(true(N1),-1));
        graphs2_blpgl(:,i) = -LL2(tril(true(N2),-1));
        
    end
    toc;
end

%% viz
% A2 = (A2+A2')/2;
A2 = squareform(graphs2_bpgl(:,20));
A2(A2<1e-4) = 0;
% A2 = squareform(graphs2_rpgl(:,1,8));
G = graph(A2);
figure(3);
plot(G,'XData',x,'YData',y,'ZData',z);
% scatter3(x,y,z);

%%
figure(10)
tcl = tiledlayout(3,5);
tcl.TileSpacing = 'compact';
tcl.Padding = 'compact';
for i = 1:5
    nexttile
    w = graphs2_rpgl(:,1,2*(i-1)+1);
    imagesc(squareform(w));
    c=colorbarpzn(0, max(w),'rev');c.FontSize=16;
    axis square
    xticks([]);yticks([]);
%     axis off
    if i == 1
        ylabel('PGL','FontSize',24)
    end
end
for i = 10:-1:6
    nexttile
    pm = reshape(graphs2_teralasso2(:,1*(i-1)+1,1*(i-1)+1),N2,N2);
    imagesc(-pm+diag(diag(pm)));%, 'AlphaData', .5);
%     clim([-0.3,0.3]);
%     colormap(redblue(64,[-0.3,0.3]))
    c=colorbarpzn(min(pm(:)), -min(pm(:)),'rev');c.FontSize=16;
    axis square
    xticks([]);yticks([]);
%     axis off
    if i == 10
        ylabel('TeraLasso','FontSize',24)
    end
end
for i = 1:5
    nexttile
    w = graphs2_blpgl(:,1*(i-1)+1);
    imagesc(squareform(w));
    c=colorbarpzn(0, max(w),'rev');c.FontSize=16;
    axis square
    xticks([]);yticks([]);
%     axis off
    if i == 1
        ylabel('MWGL (ours)','FontSize',24)
    end
end
% print('molene_laplacian_compare','-dpng','-r600')