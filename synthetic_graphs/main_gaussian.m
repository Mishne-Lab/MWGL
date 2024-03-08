clear;close all
%% set up parameters
addpath("../")
addpath("../baselines");
addpath("../utils/glmnet-matlab")
N1 = 20;
N2 = 25;
N = N1*N2;
p1 = N1*(N1-1)/2;
p2 = N2*(N2-1)/2;
upper = 2; % range of edge weights
lower = 0.1;
model = 'er'; % 'pa', 'ws', 'grid'
filter = 'gmrf';
nreplicate = 50;
Ms = [10,100,1000,10000,100000]; % number of graph signals

%% load
filename = join([model,"_N1=",num2str(N1),"_N2=",num2str(N2),"_weight=[",num2str(lower),",",num2str(upper),"].mat"],"");
load(filename)

%% set up the baselines
baselines.pst = 0;
baselines.rpgl = 0;
baselines.biglasso = 0;
baselines.teralasso = 0;
baselines.eiglasso = 0;
baselines.mwgl = 1;

beta_rpgl = 0.1.^(0:0.2:2);
beta_rpgl = [beta_rpgl,0];
len_beta_rpgl = length(beta_rpgl);

lambda_biglasso = 0.1.^[2:0.2:4];
lambda_biglasso = [lambda_biglasso,0];
len_lambda_biglasso = length(lambda_biglasso);

lambda_teralasso = 0.1.^[2:0.2:4];%0.1.^[1:0.2:3];;%0;%
lambda_teralasso = [lambda_teralasso,0];
len_lambda_teralasso = length(lambda_teralasso);

lambda_eiglasso = 0.1.^[1:0.2:3];%0.1.^[1:0.2:3];;%0;%
lambda_eiglasso = [lambda_eiglasso,0];
len_lambda_eiglasso = length(lambda_eiglasso);

alpha_mwgl = 0.1.^(1:0.2:3);
alpha_mwgl = [alpha_mwgl,0];%
len_alpha_mwgl = length(alpha_mwgl);
    

for k = 1:length(Ms)
M = Ms(k);

graphs1_pst = zeros(p1,nreplicate);
graphs1_rpgl = zeros(p1,len_beta_rpgl,len_beta_rpgl,nreplicate);
graphs1_biglasso = zeros(N1^2,len_lambda_biglasso,len_lambda_biglasso,nreplicate);
graphs1_teralasso = zeros(N1^2,len_lambda_teralasso,len_lambda_teralasso,nreplicate);
graphs1_eiglasso = zeros(N1^2,len_lambda_eiglasso,len_lambda_eiglasso,nreplicate);
graphs1_mwgl = zeros(p1,len_alpha_mwgl,nreplicate);

graphs2_pst = zeros(p2,nreplicate);
graphs2_rpgl = zeros(p2,len_beta_rpgl,len_beta_rpgl,nreplicate);
graphs2_biglasso = zeros(N2^2,len_lambda_biglasso,len_lambda_biglasso,nreplicate);
graphs2_teralasso = zeros(N2^2,len_lambda_teralasso,len_lambda_teralasso,nreplicate);
graphs2_eiglasso = zeros(N2^2,len_lambda_eiglasso,len_lambda_eiglasso,nreplicate);
graphs2_mwgl = zeros(p2,len_alpha_mwgl,nreplicate);

for ii = 1:nreplicate
% parfor (ii = 1:nreplicate,10)

%% generate or load graphs
L_0 = data{ii,2};
Lp1_0 = data{ii,4};
Lp2_0 = data{ii,6};
[X,X_noisy] = generate_signals(L_0,filter,M);
X_M = X(:,1:M);

%% main pst loop
if baselines.pst == 1
    tic;
    
    param.N1 = N1;
    param.N2 = N2;
    param.template = 'noisy';
    param.gso = 'laplacian';
    param.cnt = 1000;
    param.max_iter = 10000;
    param.max_err = 0.05;
    param.delta_err = 0.05;
    param.thr = 0;
    [A,A1,A2] = pst(X_M,param);
    L = diag(sum(A,1))-A;
    L1 = diag(sum(A1,1))-A1;
    L2 = diag(sum(A2,1))-A2;
    graphs1_pst(:,ii) = -L1(tril(true(N1),-1));
    graphs2_pst(:,ii) = -L2(tril(true(N2),-1));
    
    toc;
end

%% main rpgl loop
if baselines.rpgl == 1
    tic;
    
    for i = 1:len_beta_rpgl
        for j = 1:len_beta_rpgl
            param = struct();
            param.N1 = N1;
            param.N2 = N2;
            param.solver = 'idqp';
            param.beta1 = beta_rpgl(i);
            param.beta2 = beta_rpgl(j);
            param.rho = 0.001;
            param.tol = 1e-6;
            param.max_iter = 20000;
            param.thr = 0;
            [L,L1,L2] = rpgl(X_M,param);
            graphs1_rpgl(:,i,j,ii) = -L1(tril(true(N1),-1));
            graphs2_rpgl(:,i,j,ii) = -L2(tril(true(N2),-1));
        end
    end
    toc;
end

%% main biglasso loop
if baselines.biglasso == 1
    tic;

    X_M_rs = reshape(X_M,N2,N1,[]);
    X_M_rs = normalize(X_M_rs,2,"norm");
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(reshape(X_M,N2,N1,[]),[2,1,3]);
    X_M_rs = normalize(X_M_rs,2,"norm");
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;

    for i = 1:len_lambda_biglasso
        for j = 1:len_lambda_biglasso
            [W, W_dual, Psi, Theta] = biglasso(S,T,[lambda_biglasso(i),lambda_biglasso(j)]);
            L2 = Psi;
            L1 = Theta;
            graphs1_biglasso(:,i,j,ii) = -L1(tril(true(N1),-1));
            graphs2_biglasso(:,i,j,ii) = -L2(tril(true(N2),-1));
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
    
    tol = 1e-4;
    maxiter = 100;
    for i = 1:len_lambda_teralasso
        for j = 1:len_lambda_teralasso
            [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda_teralasso(i),lambda_teralasso(j)],maxiter);
            L1 = PsiH{1};
            L2 = PsiH{2};
            graphs1_teralasso(:,i,j,ii) = -L1(tril(true(N1),-1));
            graphs2_teralasso(:,i,j,ii) = -L2(tril(true(N2),-1));
        end
    end
    toc;
end

%% main eiglasso loop
if baselines.eiglasso == 1
    tic;

    X_M_rs = reshape(X_M,N2,N1,[]);
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(X_M_rs,[2,1,3]);
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;
    
    tol = 1e-4;
    maxiter = 100;
    for i = 1:len_lambda_eiglasso
        for j = 1:len_lambda_eiglasso
            [Theta, Psi, ts, fs] = eiglasso_joint(S, T, lambda_eiglasso(i), lambda_eiglasso(j));
            L1 = Theta+Theta'-diag(diag(Theta));
            L2 = Psi+Psi'-diag(diag(Psi));
            graphs1_eiglasso(:,i,j,ii) = -L1(tril(true(N1),-1));
            graphs2_eiglasso(:,i,j,ii) = -L2(tril(true(N2),-1));
        end
    end
    toc;
end

%% main mwgl loop
if baselines.mwgl == 1
    tic;
    
%     parfor (i = 1:len_alpha, 10)
    for i = 1:len_alpha_mwgl
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha_mwgl(i)*N2,alpha_mwgl(i)*N1];
        param.inv_compute = 'eig';
        param.max_iter = 5000;
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = mwgl(X_M,param);
        graphs1_mwgl(:,i,ii) = -L1(tril(true(N1),-1));
        graphs2_mwgl(:,i,ii) = -L2(tril(true(N2),-1));
        
    end
    toc;
end

end

if baselines.pst == 1
    res1_pst_graphs1(:,:,k) = graphs1_pst;
    res2_pst_graphs2(:,:,k) = graphs2_pst;
end
if baselines.rpgl == 1
    res1_rpgl_graphs1(:,:,:,:,k) = graphs1_rpgl;
    res2_rpgl_graphs2(:,:,:,:,k) = graphs2_rpgl;
end
if baselines.bpgl == 1
    res1_bpgl_graphs1(:,:,:,k) = graphs1_bpgl;
    res2_bpgl_graphs2(:,:,:,k) = graphs2_bpgl;
end
if baselines.biglasso == 1
    res1_biglasso_graphs1(:,:,:,:,k) = graphs1_biglasso;
    res2_biglasso_graphs2(:,:,:,:,k) = graphs2_biglasso;
end
if baselines.teralasso == 1
    res1_teralasso_graphs1(:,:,:,:,k) = graphs1_teralasso;
    res2_teralasso_graphs2(:,:,:,:,k) = graphs2_teralasso;
end

if baselines.eiglasso == 1
    res1_eiglasso_graphs1(:,:,:,:,k) = graphs1_eiglasso;
    res2_eiglasso_graphs2(:,:,:,:,k) = graphs2_eiglasso;
end
if baselines.mwgl == 1
    res1_mwgl_graphs1(:,:,:,k) = graphs1_mwgl;
    res2_mwgl_graphs2(:,:,:,k) = graphs2_mwgl;
end

end