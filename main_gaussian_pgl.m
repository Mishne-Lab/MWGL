% clear;close all
% load('data/weighted_prod_graphs_03er10_03er15_w301.mat');
% % load('data/weighted_prod_graphs_1ba10_1ba15_w301.mat');
% % load('data/weighted_prod_graphs_201ws10_201ws15_w301.mat');

% load('data/weighted_prod_graphs_tr=N_03er10_03er15_w201.mat');
% load('data/weighted_prod_graphs_tr=N_1ba10_1ba15_w301.mat');
% load('data/weighted_prod_graphs_tr=N_201ws10_201ws15_w301.mat');
%% Generate a graph
N1 = 10;
N2 = 15;
N = N1*N2;
nreplicate = 20; % repeat the same experiment (based on different graphs)
mmetric = 3; % use f-score as main metric to determine the best parameter
Ms = [10,100,1000,10000,100000];
% Ms = [10];
res_graphs = zeros(N*(N-1)/2,nreplicate,length(Ms));
res_graphs1 = zeros(N1*(N1-1)/2,nreplicate,length(Ms));
res_graphs2 = zeros(N2*(N2-1)/2,nreplicate,length(Ms));

for k = 1:length(Ms)
M = Ms(k)
Xs = [data{:,7}];
% Xs = [data{:,8}];
% Xs = Xs + 0.1*randn(size(Xs));
L0s = [data{:,2}];
Xs = reshape(Xs,N1*N2,[],nreplicate);
Xs = Xs(:,1:M,:);
Xs = permute(Xs,[3,1,2]);
L0s = reshape(L0s,N,N,nreplicate);
L0s = permute(L0s,[3,1,2]);

Lp1_0s = [data{:,4}];
Lp2_0s = [data{:,6}];
Lp1_0s = reshape(Lp1_0s,N1,N1,nreplicate);
Lp1_0s = permute(Lp1_0s,[3,1,2]);
Lp2_0s = reshape(Lp2_0s,N2,N2,nreplicate);
Lp2_0s = permute(Lp2_0s,[3,1,2]);

graphs = zeros(N*(N-1)/2,nreplicate);
graphs1 = zeros(N1*(N1-1)/2,nreplicate);
graphs2 = zeros(N2*(N2-1)/2,nreplicate);

parfor (ii = 1:nreplicate,10)
% for ii = 1:nreplicate


%%
% % [A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.75,0.5);
% [Ap1,XCoords, YCoords] = construct_graph(N1,'er',0.3);
% % [A,XCoords, YCoords] = construct_graph(param.N,'pa',1);
% 
% % [A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.75,0.5);
% [Ap2,XCoords, YCoords] = construct_graph(N2,'er',0.3);
% % [A,XCoords, YCoords] = construct_graph(param.N,'pa',1);

%% Generate the graph Laplacian 
% Lp1_0 = full(sgwt_laplacian(Ap1,'opt','raw'));
% Ap1_0 = -Lp1_0+diag(diag(Lp1_0));
% W = triu(rand(N1)*2.9 + 0.1, 1);
% Ap1_0 = Ap1_0 .* (W+W');
% Lp1_0 = diag(sum(Ap1_0)) - Ap1_0;
% % L_0 = L_0/trace(L_0)*param.N;
% 
% Lp2_0 = full(sgwt_laplacian(Ap2,'opt','raw'));
% Ap2_0 = -Lp2_0+diag(diag(Lp2_0));
% W = triu(rand(N2)*2.9 + 0.1, 1);
% Ap2_0 = Ap2_0 .* (W+W');
% Lp2_0 = diag(sum(Ap2_0)) - Ap2_0;
% 
% % % tensor product
% % L_0 = kron(Lp1_0, Lp2_0);
% 
% % cartesian product
% L_0 = kron(Lp1_0,eye(N2)) + kron(eye(N1),Lp2_0);
% A_0 = -L_0+diag(diag(L_0));

%% generate training signals
% [V,D] = eig(full(L_0));
% sigma = pinv(D);
% mu = zeros(1,N1*N2);
% num_of_signal = 10000;
% gftcoeff = mvnrnd(mu,sigma,num_of_signal);
% X = V*gftcoeff';
% X_noisy = X + 0.5*randn(size(X));

%%
% L_0 = data{ii,2};
L_0 = squeeze(L0s(ii,:,:));
Lp1_0 = squeeze(Lp1_0s(ii,:,:));
Lp2_0 = squeeze(Lp2_0s(ii,:,:));
% X = data{ii,7};
% X = Xs(:,:,ii);
X = squeeze(Xs(ii,:,:));

%% set parameters
param = struct();
param.N1 = N1;
param.N2 = N2;
param.reg = 'l1';%'reweighted-l1';%
param.w_solver = 'gd';
param.w_init = 'naive';
param.Ui_init = 'L_sep';%'L_nks';
param.tr_normalize = 0;
param.max_iter = 500000;%0;%
param.reltol = 1e-6;%1e-7;%
param.abstol = 0; % 1e-6;
param.eps = 1e-4;%0.5;%
param.k = 1;
param.c1 = 0;
param.c2 = 32;
% alpha = 10.^[-1:-0.1:-3];
% beta = 10.^[0:-0.1:-2];
% lambda = 10.^[3:-0.05:0];
alpha = 0;%1/M;%0;%5e-5;%0.1;%
% beta = 1.5;
% gamma = 1.5;
% eta = 1.5;
beta = 10;%5;%
gamma = 10;%5;%
eta = 10;%5;%

% new
beta = 1000;
gamma = 100;

% precision = zeros(length(alpha),length(beta));
% recall = zeros(length(alpha),length(beta));
% Fmeasure = zeros(length(alpha),length(beta));
% NMI = zeros(length(alpha),length(beta));
% num_of_edges = zeros(length(alpha),length(beta));

%% main loop
tic;
for i = 1:length(alpha)
    for j = 1:length(beta)
        param.alpha = alpha(i);
        param.beta = beta(j);
        param.gamma = gamma;
        param.eta = eta;
        param.step_size = 1e-2;
        % GL-SigRep
%         [L,L1,L2,nll,obj] = pgl(X,param);
        [L,L1,L2,nll,obj] = cpgl2(X,param);
        LL = L;
        LL1 = L1;
        LL2 = L2;
%         [X,Y,T,AUC] = perfcurve(L_0(tril(true(N),-1))<0,-L(tril(true(N),-1)),1);
        L(abs(L)<0.1)=0;
        [precision,recall,Fmeasure,NMI,num_of_edges] = graph_learning_perf_eval(L_0,L);
        L1(abs(L1)<0.1)=0;
        [precision1,recall1,Fmeasure1,NMI1,num_of_edges1] = graph_learning_perf_eval(Lp1_0,L1);
        L2(abs(L2)<0.1)=0;
        [precision2,recall2,Fmeasure2,NMI2,num_of_edges2] = graph_learning_perf_eval(Lp2_0,L2);
    end
end
toc;

%% performance
% result = zeros(5,1);
% result(1) = precision;
% result(2) = recall;
% result(3) = Fmeasure;
% result(4) = NMI;
% result(5) = num_of_edges;
result = [precision,recall,Fmeasure,NMI,num_of_edges];
result1 = [precision1,recall1,Fmeasure1,NMI1,num_of_edges1];
result2 = [precision2,recall2,Fmeasure2,NMI2,num_of_edges2];
res(:,ii) = result;
res1(:,ii) = result1;
res2(:,ii) = result2;

% data{ii,1} = A_0;
% data{ii,2} = L_0;
% data{ii,3} = Ap1_0;
% data{ii,4} = Lp1_0;
% data{ii,5} = Ap2_0;
% data{ii,6} = Lp2_0;
% data{ii,7} = X;
% data{ii,8} = X_noisy;

graphs(:,ii) = -LL(tril(true(N1*N2),-1));
graphs1(:,ii) = -LL1(tril(true(N1),-1));
graphs2(:,ii) = -LL2(tril(true(N2),-1));

end

res_best(:,:,k) = evaluate(res,mmetric);
res_graphs(:,:,k) = graphs;
res_best1(:,:,k) = evaluate(res1,mmetric);
res_graphs1(:,:,k) = graphs1;
res_best2(:,:,k) = evaluate(res2,mmetric);
res_graphs2(:,:,k) = graphs2;
end

%% plot
metric_names = ["precision","recall","f-score","NMI"];
% sgtitle('Cartesian Product of ER10 & ER15');
for j = 1:4
%     figure(j);
    subplot(2,2,j);
    hold all;
    errorbar(log(Ms),squeeze(res_best(j,1,:)),squeeze(res_best(j,2,:)));ylim([0,1]);
%     xlabel("log(#) of signals");ylabel(metric_names(j));
%     legend('pst','rpgl','bpgl', 'ours','Location','southeast');
    legend('ours','Location','southeast');
end
