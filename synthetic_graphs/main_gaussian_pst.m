clear;close all
% load('data/weighted_prod_graphs_tr=N_03er10_03er15_w201.mat');
% load('data/weighted_prod_graphs_1ba10_1ba15_w301.mat');
% load('data/weighted_prod_graphs_201ws10_201ws15_w301.mat');

% load('data/weighted_prod_graphs_tr=N_03er10_03er15_w301.mat');
% load('data/weighted_strong_prod_graphs_tr=N_1ba10_1ba15_w305.mat');
% load("data/weighted_strong_prod_graphs_tr=N_03er10_03er15_w305.mat");
% load("data/weighted_strong_prod_graphs_tr=N_201ws10_201ws15_w305.mat");

% load('data/weighted_tensor_prod_graphs_tr=N_03er10_03er15_w21.mat');
% load('data/weighted_tensor_prod_graphs_tr=N_1ba10_1ba15_w21.mat');
% load('data/weighted_tensor_prod_graphs_03er10_03er15_w305.mat');

load("data/new_weighted_prod_graphs_wosignals_2ba20_2ba25_w201.mat")
% load("data/new_weighted_prod_graphs_wosignals_03er32_03er32_w201.mat")
%% Generate a graph
nreplicate = 100; % repeat the same experiment (based on different graphs)

baselines.pst = 0;
baselines.rpgl = 0;
baselines.bpgl = 0;
baselines.blpgl = 0;
baselines.biglasso = 0;
baselines.teralasso = 0;
baselines.eiglasso = 1;
mask_and_impute = 0;
load_saved = 0;
eval = 1;
filter = 'gmrf';
pd_type = 'cartesian';

N1 = 10;
N2 = 15;

N1 = 32;
N2 = 32;

N1 = 20;
N2 = 25;

N = N1*N2;

p1 = N1*(N1-1)/2;
p2 = N2*(N2-1)/2;

% Ms = [1000];
Ms = [10,100,1000,10000,100000];

mmetric = 9; % use f-score as main metric to determine the best parameter

for k = 1:length(Ms)
M = Ms(k);

% L0s = [data{:,2}];
% L0s = reshape(L0s,N,N,nreplicate);
% L0s = permute(L0s,[3,1,2]);
% Lp1_0s = [data{:,4}];
% Lp2_0s = [data{:,6}];
% Lp1_0s = reshape(Lp1_0s,N1,N1,nreplicate);
% Lp1_0s = permute(Lp1_0s,[3,1,2]);
% Lp2_0s = reshape(Lp2_0s,N2,N2,nreplicate);
% Lp2_0s = permute(Lp2_0s,[3,1,2]);

graphs1_pst = zeros(p1,nreplicate);
graphs1_rpgl = zeros(p1,12,12,nreplicate);
graphs1_bpgl = zeros(p1,17,nreplicate);
graphs1_blpgl = zeros(p1,nreplicate);
graphs1_teralasso = zeros(p1,12,12,nreplicate);
graphs1_eiglasso = zeros(p1,12,12,nreplicate);

graphs2_pst = zeros(p2,nreplicate);
graphs2_rpgl = zeros(p2,12,12,nreplicate);
graphs2_bpgl = zeros(p2,17,nreplicate);
graphs2_blpgl = zeros(p2,nreplicate);
graphs2_teralasso = zeros(p2,12,12,nreplicate);
graphs2_eiglasso = zeros(p2,12,12,nreplicate);

for ii = 1:nreplicate
% parfor (ii = 1:nreplicate,10)

%% generate or load graphs
try
    L_0 = data{ii,2};
    Lp1_0 = data{ii,4};
    Lp2_0 = data{ii,6};
%     L_0 = squeeze(L0s(ii,:,:));
%     Lp1_0 = squeeze(Lp1_0s(ii,:,:));
%     Lp2_0 = squeeze(Lp2_0s(ii,:,:));
    [X,X_noisy] = generate_graph_signals(L_0,filter,M);
%     X = data{ii,7};
%     X_noisy = data{ii,8};
%     if size(X,2)<M
%         [X_new,X_noisy_new] = generate_graph_signals(L_0,filter,M-size(X,2));
%         X = [X,X_new];
%         X_noisy = [X_noisy,X_noisy_new];
%         data{ii,7} = X;
%         data{ii,8} = X_noisy;
%     end

catch
    while true
%         [Ap1,XCoords, YCoords] = construct_graph(N1,'gaussian',0.75,0.5);
%         [Ap1,XCoords, YCoords] = construct_graph(N1,'er',0.3);
%         [Ap1,XCoords, YCoords] = construct_graph(N1,'pa',2);
%         [Ap1,XCoords, YCoords] = construct_graph(N1,'ws',3,0.1);
        [Ap1,XCoords, YCoords] = construct_graph(N1,'grid',4,5);
        if all(conncomp(graph(Ap1))==1)
            break;
        end
    end
    
    while true
%         [Ap2,XCoords, YCoords] = construct_graph(N2,'gaussian',0.75,0.5);
%         [Ap2,XCoords, YCoords] = construct_graph(N2,'er',0.3);
%         [Ap2,XCoords, YCoords] = construct_graph(N2,'pa',2);
%         [Ap2,XCoords, YCoords] = construct_graph(N2,'ws',3,0.1);
        [Ap2,XCoords, YCoords] = construct_graph(N2,'grid',5,5);
        if all(conncomp(graph(Ap2))==1)
            break;
        end
    end
    
    %% Generate the graph Laplacian 
    Lp1_0 = full(sgwt_laplacian(Ap1,'opt','raw'));
    Ap1_0 = -Lp1_0+diag(diag(Lp1_0));
    W = triu(rand(N1)*1.9 + 0.1, 1);
%     W = triu(rand(N1)*1 + 1, 1);
    Ap1_0 = Ap1_0 .* (W+W');
    Lp1_0 = diag(sum(Ap1_0,1)) - Ap1_0;
%     Ap1_0 = Ap1_0/trace(Lp1_0)*N1;
%     Lp1_0 = Lp1_0/trace(Lp1_0)*N1;
%     Ap1_0 = diag(1./sqrt(diag(Lp1_0)))*Ap1_0*diag(1./sqrt(diag(Lp1_0)));
%     Lp1_0 = diag(1./sqrt(diag(Lp1_0)))*Lp1_0*diag(1./sqrt(diag(Lp1_0)));
    
    Lp2_0 = full(sgwt_laplacian(Ap2,'opt','raw'));
    Ap2_0 = -Lp2_0+diag(diag(Lp2_0));
    W = triu(rand(N2)*1.9 + 0.1, 1);
%     W = triu(rand(N2)*1 + 1, 1);
    Ap2_0 = Ap2_0 .* (W+W');
    Lp2_0 = diag(sum(Ap2_0,1)) - Ap2_0;
%     Ap2_0 = Ap2_0/trace(Lp2_0)*N2;
%     Lp2_0 = Lp2_0/trace(Lp2_0)*N2;
%     Ap2_0 = diag(1./sqrt(diag(Lp2_0)))*Ap2_0*diag(1./sqrt(diag(Lp2_0)));
%     Lp2_0 = diag(1./sqrt(diag(Lp2_0)))*Lp2_0*diag(1./sqrt(diag(Lp2_0)));
    
    switch pd_type
        case 'cartesian'
            % cartesian product
            L_0 = kron(Lp1_0,eye(N2)) + kron(eye(N1),Lp2_0);
            % L_0 = L_0/trace(L_0)*N1*N2;
            A_0 = -L_0+diag(diag(L_0));
        case 'tensor'
            % tensor product
            A_0 = kron(Ap1_0,Ap2_0);
            L_0 = diag(sum(A_0))-A_0;
        case 'strong'
            % strong product
            A_0 = kron(Ap1_0,Ap2_0)+kron(Ap1_0,eye(N2))+kron(eye(N1),Ap2_0);
            L_0 = diag(sum(A_0))-A_0;
    end
    
    %% generate training signals
    num_of_signal = max(Ms);
    [X,X_noisy] = generate_graph_signals(L_0,filter,num_of_signal);

    data{ii,1} = A_0;
    data{ii,2} = L_0;
    data{ii,3} = Ap1_0;
    data{ii,4} = Lp1_0;
    data{ii,5} = Ap2_0;
    data{ii,6} = Lp2_0;
%     data{ii,7} = X;
%     data{ii,8} = X_noisy;
end
X_M = X(:,1:M);
% X_M = X_noisy(:,1:M);
% X_M = X_M + 0.1*randn(size(X_M));

if mask_and_impute
    mask1 = zeros(N1,1);
    mask2 = zeros(N2,1);
    mask1(1:end-5) = 1;
    mask2(1:end-5) = 1;
    mask = 1-kron(1-mask1,1-mask2);
    X_M(~mask,:) = ones(25,1)*mean(X_M(mask>0,:),1);
end

%% main pst loop
if baselines.pst == 1
    tic;
    
    param.N1 = N1;
    param.N2 = N2;
    param.template = 'noisy';
    param.pd_type = pd_type;%'tensor';%'strong';%'cartesian';%
    param.gso = 'laplacian';
    % param.gso = 'adjacency';
    param.cnt = 1000;
    param.max_iter = 10000;
    param.max_err = 0.05;
    param.delta_err = 0.05;
    param.thr = 0;
%     param.thr = 0.01;
%     param.thr = 0.1;
    [A,A1,A2] = pst(X_M,param);
    L = diag(sum(A,1))-A;
    L1 = diag(sum(A1,1))-A1;
    L2 = diag(sum(A2,1))-A2;
    LL1 = L1;
    LL2 = L2;
%     L(abs(L)<0.1)=0;
    L(abs(L)<1e-4)=0;
    L1(abs(L1)<1e-4)=0;
    L2(abs(L2)<1e-4)=0;
%     [precision_pst,recall_pst,Fmeasure_pst,NMI_pst,num_of_edges_pst] = graph_learning_perf_eval(L_0,L);
%     res_pst(1,ii) = precision_pst;
%     res_pst(2,ii) = recall_pst;
%     res_pst(3,ii) = Fmeasure_pst;
%     res_pst(4,ii) = NMI_pst;
%     res_pst(5,ii) = num_of_edges_pst;
    [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
    if eval == 1
        res_pst(:,ii) = res;
        res1_pst(:,ii) = res1;
        res2_pst(:,ii) = res2;
    end
    graphs1_pst(:,ii) = -LL1(tril(true(N1),-1));
    graphs2_pst(:,ii) = -LL2(tril(true(N2),-1));
    
    toc;
end

%% main rpgl loop
if baselines.rpgl == 1
    tic;
    
    assert(pd_type=="cartesian");
%     old
    beta1 = 0.1.^(0:0.2:2);
    beta2 = 0.1.^(0:0.2:2);
% %     new
%     beta1 = 0.1.^(-1:0.2:1);
%     beta2 = 0.1.^(-1:0.2:1);
    beta1 = [beta1,0];
    beta2 = [beta2,0];
    % param.beta1 = 0.25;
    % param.beta2 = 0.25;
    len_beta1 = length(beta1);
    len_beta2 = length(beta2);
    
    % precision_rpgl = zeros(length(beta1),length(beta2));
%     parfor (i = 1:len_beta1, len_beta1)
    for i = 1:len_beta1
        for j = 1:len_beta2
            param = struct();
            param.N1 = N1;
            param.N2 = N2;
            param.solver = 'idqp';
            param.beta1 = beta1(i);
            param.beta2 = beta2(j);
%             param.rho = 0.00005;
            param.rho = 0.001;
            param.tol = 1e-6;
            param.max_iter = 20000;
            param.thr = 0;
%             param.thr = 0.1;
            [L,L1,L2] = rpgl(X_M,param);
            LL1 = L1;
            LL2 = L2;
            % L(abs(L)<0.1)=0;
            L(abs(L)<10^(-4))=0;
            L1(abs(L1)<0.1)=0;
            L2(abs(L2)<0.1)=0;
%             [precision_rpgl(i,j),recall_rpgl(i,j),Fmeasure_rpgl(i,j),NMI_rpgl(i,j),num_of_edges_rpgl(i,j)] = graph_learning_perf_eval(L_0,L);
            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            if eval == 1
                res_rpgl(i,j,:,ii) = res;
                res1_rpgl(i,j,:,ii) = res1;
                res2_rpgl(i,j,:,ii) = res2;
            end
            graphs1_rpgl(:,i,j,ii) = -LL1(tril(true(N1),-1));
            graphs2_rpgl(:,i,j,ii) = -LL2(tril(true(N2),-1));
        end
    end
%     res_rpgl(:,:,1,ii) = precision_rpgl;
%     res_rpgl(:,:,2,ii) = recall_rpgl;
%     res_rpgl(:,:,3,ii) = Fmeasure_rpgl;
%     res_rpgl(:,:,4,ii) = NMI_rpgl;
%     res_rpgl(:,:,5,ii) = num_of_edges_rpgl
    
    toc;
end

%% main bpgl loop
if baselines.bpgl == 1
    tic;
    
%     alpha = 0.75.^(10:29)/sqrt(M); % originally 0:15
    alpha = 0.75.^(0:15)*sqrt(log(N1*N2)/M);
    alpha = [alpha,0];
    len_alpha = length(alpha);
    
%     parfor (i = 1:len_alpha, 10)
    for i = 1:len_alpha
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i); % *log(N1*N2);
        param.rho = 0.75/log(M);
        param.pd_type = pd_type;%'tensor';%'strong';%'cartesian';%
        param.max_iter = 1000;
        param.tol = 1e-6;
        param.thr = 0;
%         param.thr = 0.01;
%         param.thr = 0.1;
        [L,L1,L2] = bpgl(X_M,param);
        LL1 = L1;
        LL2 = L2;
        L(abs(L)<10^(-4))=0;
    %     L(abs(L)<0.1)=0;
        L1(abs(L1)<0.1)=0;
        L2(abs(L2)<0.1)=0;
    %     L = kron(L1,eye(param.N2))+kron(eye(param.N1),L2);
%         [precision_bpgl(i),recall_bpgl(i),Fmeasure_bpgl(i),NMI_bpgl(i),num_of_edges_bpgl(i)] = graph_learning_perf_eval(L_0,L);
        [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
        if eval == 1
            res_bpgl(i,:,ii) = res;
            res1_bpgl(i,:,ii) = res1;
            res2_bpgl(i,:,ii) = res2;
        end
        graphs1_bpgl(:,i,ii) = -LL1(tril(true(N1),-1));
        graphs2_bpgl(:,i,ii) = -LL2(tril(true(N2),-1));
        
    end
%     res_bpgl(:,1,ii) = precision_bpgl;
%     res_bpgl(:,2,ii) = recall_bpgl;
%     res_bpgl(:,3,ii) = Fmeasure_bpgl;
%     res_bpgl(:,4,ii) = NMI_bpgl;
%     res_bpgl(:,5,ii) = num_of_edges_bpgl;
    
    toc;
end

%% main blpgl loop
if baselines.blpgl == 1
    tic;
    
    alpha = 0;%0.1.^(1:0.2:3);
    len_alpha = length(alpha);
    
%     parfor (i = 1:len_alpha, 10)
    for i = 1:len_alpha
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i);
        param.pd_type = pd_type;%'cartesian';%'tensor';%'strong';%
        param.inv_compute = 'eig';
        param.max_iter = 5000;
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = blpgl(X_M,param);
        LL1 = L1;
        LL2 = L2;
        L(abs(L)<10^(-4))=0;
    %     L(abs(L)<0.1)=0;
        L1(abs(L1)<0.1)=0;
        L2(abs(L2)<0.1)=0;
    %     L = kron(L1,eye(param.N2))+kron(eye(param.N1),L2);
%         [precision_bpgl(i),recall_bpgl(i),Fmeasure_bpgl(i),NMI_bpgl(i),num_of_edges_bpgl(i)] = graph_learning_perf_eval(L_0,L);
        [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
        if eval == 1
            res_blpgl(i,:,ii) = res;
            res1_blpgl(i,:,ii) = res1;
            res2_blpgl(i,:,ii) = res2;
        end
        graphs1_blpgl(:,i,ii) = -LL1(tril(true(N1),-1));
        graphs2_blpgl(:,i,ii) = -LL2(tril(true(N2),-1));
        
    end
%     res_bpgl(:,1,ii) = precision_bpgl;
%     res_bpgl(:,2,ii) = recall_bpgl;
%     res_bpgl(:,3,ii) = Fmeasure_bpgl;
%     res_bpgl(:,4,ii) = NMI_bpgl;
%     res_bpgl(:,5,ii) = num_of_edges_bpgl;
    
    toc;
end

%% main biglasso loop
if baselines.biglasso == 1
    tic;

    assert(pd_type=="cartesian");

    X_M_rs = reshape(X_M,N2,N1,[]);
    X_M_rs = normalize(X_M_rs,2,"norm");
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(reshape(X_M,N2,N1,[]),[2,1,3]);
    X_M_rs = normalize(X_M_rs,2,"norm");
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;

%     X_M_rs = reshape(X_M,N2,[]);
%     X_M_rs = normalize(X_M_rs,2,"norm");
%     T = X_M_rs*X_M_rs'/N1/M;
%     X_M_rs = reshape(permute(reshape(X_M,N2,N1,[]),[2,1,3]),N1,[]);
%     X_M_rs = normalize(X_M_rs,2,"norm");
%     S = X_M_rs*X_M_rs'/N2/M;

    lambda = 0;%0.1.^[1:0.2:3];%
    len_lambda = length(lambda);
    for i = 1:len_lambda
        for j = 1:len_lambda
            [W, W_dual, Psi, Theta] = biglasso(S,T,[lambda(i),lambda(j)]);
            LL2 = Psi;
            LL1 = Theta;
            Psi(Psi>0) = 0;
            Theta(Theta>0) = 0;
            L1 = -diag(sum(Theta,1))+Theta;
            L2 = -diag(sum(Psi,1))+Psi;
            L = kron(L1,eye(N2))+kron(eye(N1),L2);
            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            if eval == 1
                res_biglasso(i,j,:,ii) = res;
                res1_biglasso(i,j,:,ii) = res1;
                res2_biglasso(i,j,:,ii) = res2;
            end
            graphs1_biglasso(:,i,j,ii) = -LL1(tril(true(N1),-1));
            graphs2_biglasso(:,i,j,ii) = -LL2(tril(true(N2),-1));
        end
    end
    toc;
end

%% main teralasso loop
if baselines.teralasso == 1
    tic;
    
    assert(pd_type=="cartesian");

    X_M_rs = reshape(X_M,N2,N1,[]);
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(X_M_rs,[2,1,3]);
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;
    lambda = 0.1.^[2:0.2:4];%0.1.^[1:0.2:3];;%0;%
    lambda = [lambda,0];
    len_lambda = length(lambda);
    tol = 1e-4;
    maxiter = 100;
    for i = 1:len_lambda
        for j = 1:len_lambda
            [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda(i),lambda(j)],maxiter);
            LL1 = PsiH{1};
            LL2 = PsiH{2};
            L1 = PsiH{1};
            L2 = PsiH{2};
            L1(L1>0) = 0;
            L2(L2>0) = 0;
            L1 = -diag(sum(L1,1))+L1;
            L2 = -diag(sum(L2,1))+L2;
            L = kron(L1,eye(N2))+kron(eye(N1),L2);
            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            if eval == 1
                res_teralasso(i,j,:,ii) = res;
                res1_teralasso(i,j,:,ii) = res1;
                res2_teralasso(i,j,:,ii) = res2;
            end
            graphs1_teralasso(:,i,j,ii) = -LL1(tril(true(N1),-1));
            graphs2_teralasso(:,i,j,ii) = -LL2(tril(true(N2),-1));
        end
    end
    toc;
end

%% main eiglasso loop
if baselines.eiglasso == 1
    tic;
    
    assert(pd_type=="cartesian");

    X_M_rs = reshape(X_M,N2,N1,[]);
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1;
    X_M_rs = permute(X_M_rs,[2,1,3]);
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2;
    lambda = 0.1.^[1:0.2:3];%0.1.^[1:0.2:3];;%0;%
    lambda = [lambda,0];
    len_lambda = length(lambda);
    tol = 1e-4;
    maxiter = 100;
    for i = 1:len_lambda
        for j = 1:len_lambda
            [Theta, Psi, ts, fs] = eiglasso_joint(S, T, lambda(i), lambda(j));
            LL1 = Theta+Theta'-diag(diag(Theta));
            LL2 = Psi+Psi'-diag(diag(Psi));
            L1 = LL1;
            L2 = LL2;
            L1(L1>0) = 0;
            L2(L2>0) = 0;
            L1 = -diag(sum(L1,1))+L1;
            L2 = -diag(sum(L2,1))+L2;
            L = kron(L1,eye(N2))+kron(eye(N1),L2);
            [res,res1,res2] = prod_graph_learning_perf_eval(L_0,L,Lp1_0,L1,Lp2_0,L2);
            if eval == 1
                res_eiglasso(i,j,:,ii) = res;
                res1_eiglasso(i,j,:,ii) = res1;
                res2_eiglasso(i,j,:,ii) = res2;
            end
            graphs1_eiglasso(:,i,j,ii) = -LL1(tril(true(N1),-1));
            graphs2_eiglasso(:,i,j,ii) = -LL2(tril(true(N2),-1));
        end
    end
    toc;
end

end

%% performance

if eval == 1
    if baselines.pst == 1
        res_pst_best(:,:,k) = evaluate(res_pst,mmetric);
        res1_pst_best(:,:,k) = evaluate(res1_pst,mmetric);
        res2_pst_best(:,:,k) = evaluate(res2_pst,mmetric);
        res1_pst_graphs1(:,:,k) = graphs1_pst;
        res2_pst_graphs2(:,:,k) = graphs2_pst;
    end
    if baselines.rpgl == 1
        res_rpgl_best(:,:,k) = evaluate(res_rpgl,mmetric);
        res1_rpgl_best(:,:,k) = evaluate(res1_rpgl,mmetric);
        res2_rpgl_best(:,:,k) = evaluate(res2_rpgl,mmetric);
        res1_rpgl_graphs1(:,:,:,:,k) = graphs1_rpgl;
        res2_rpgl_graphs2(:,:,:,:,k) = graphs2_rpgl;
    end
    if baselines.bpgl == 1
        res_bpgl_best(:,:,k) = evaluate(res_bpgl,mmetric);
        res1_bpgl_best(:,:,k) = evaluate(res1_bpgl,mmetric);
        res2_bpgl_best(:,:,k) = evaluate(res2_bpgl,mmetric);
        res1_bpgl_graphs1(:,:,:,k) = graphs1_bpgl;
        res2_bpgl_graphs2(:,:,:,k) = graphs2_bpgl;
    end
    if baselines.blpgl == 1
        res_blpgl_best(:,:,k) = evaluate(res_blpgl,mmetric);
        res1_blpgl_best(:,:,k) = evaluate(res1_blpgl,mmetric);
        res2_blpgl_best(:,:,k) = evaluate(res2_blpgl,mmetric);
        res1_blpgl_graphs1(:,:,:,k) = graphs1_blpgl;
        res2_blpgl_graphs2(:,:,:,k) = graphs2_blpgl;
    end
    if baselines.biglasso == 1
        res_biglasso_best(:,:,k) = evaluate(res_biglasso,mmetric);
        res1_biglasso_best(:,:,k) = evaluate(res1_biglasso,mmetric);
        res2_biglasso_best(:,:,k) = evaluate(res2_biglasso,mmetric);
        res1_biglasso_graphs1(:,:,:,:,k) = graphs1_biglasso;
        res2_biglasso_graphs2(:,:,:,:,k) = graphs2_biglasso;
    end
    if baselines.teralasso == 1
        res_teralasso_best(:,:,k) = evaluate(res_teralasso,mmetric);
        res1_teralasso_best(:,:,k) = evaluate(res1_teralasso,mmetric);
        res2_teralasso_best(:,:,k) = evaluate(res2_teralasso,mmetric);
        res1_teralasso_graphs1(:,:,:,:,k) = graphs1_teralasso;
        res2_teralasso_graphs2(:,:,:,:,k) = graphs2_teralasso;
    end

    if baselines.eiglasso == 1
        res_eiglasso_best(:,:,k) = evaluate(res_eiglasso,mmetric);
        res1_eiglasso_best(:,:,k) = evaluate(res1_eiglasso,mmetric);
        res2_eiglasso_best(:,:,k) = evaluate(res2_eiglasso,mmetric);
        res1_eiglasso_graphs1(:,:,:,:,k) = graphs1_eiglasso;
        res2_eiglasso_graphs2(:,:,:,:,k) = graphs2_eiglasso;
    end
end

end
%% plot
metric_names = ["precision","recall","f-score","NMI"];
sgtitle('Cartesian Product of ER10 & ER15 with Missing');
for j = 1:4
%     figure(j);
    subplot(2,2,j);
%     errorbar(log(Ms),squeeze(res_pst_best(j,1,:)),squeeze(res_pst_best(j,2,:)));ylim([0,1]);hold on;
%     errorbar(log(Ms),squeeze(res_rpgl_best(j,1,:)),squeeze(res_rpgl_best(j,2,:)));ylim([0,1]);hold on;
    errorbar(log(Ms),squeeze(res_bpgl_best(j,1,:)),squeeze(res_bpgl_best(j,2,:)));ylim([0,1]);hold off;
    xlabel("log(#) of signals");ylabel(metric_names(j));
    legend('pst','rpgl','bpgl','Location','southeast');
end