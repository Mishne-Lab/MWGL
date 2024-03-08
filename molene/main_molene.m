clear;close all
%% Load molene dataset

load("meteo_molene_t.mat");
X = value;
x = info{4}; y = info{3}; z = info{5}; coords = [x,y,5*z];
N2 = size(X,1);
N1 = 24;
X = reshape(X,[],24,31);
X = X(:,:,1:30);
X = X-mean(X,[1,2]);
X = reshape(X(:,:,1:30),[],30);
X = X/std(X(:));
M = size(X,2);

X_M = X;

baselines.rpgl = 1;
baselines.teralasso = 1;
baselines.mwgl = 1;
%% main rpgl loop
if baselines.rpgl == 1
    tic;
    
    beta_rpgl = 0.1.^(0:0.2:2);
    beta_rpgl = [beta_rpgl,0];
    len_beta_rpgl = length(beta_rpgl);
    
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
            param.thr = 0.1;
            [L,L1,L2] = rpgl(X_M,param);
            graphs1_rpgl(:,i,j) = -L1(tril(true(N1),-1));
            graphs2_rpgl(:,i,j) = -L2(tril(true(N2),-1));
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
    lambda_teralasso = 0.1.^[-0.5:0.1:0.5];%0;%
    lambda_teralasso = [lambda_teralasso,0];
    len_lambda_teralasso = length(lambda_teralasso);
    tol = 1e-7;
    maxiter = 10000;
    for i = 1:12
        for j = 1:12
            [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda_teralasso(i),lambda_teralasso(j)],maxiter);
            L1 = PsiH{1};
            L2 = PsiH{2};
            graphs1_teralasso(:,i,j) = L1(:);
            graphs2_teralasso(:,i,j) = L2(:);
        end
    end
    toc;
end

%% main mwgl loop
if baselines.mwgl == 1
    tic;
    
    alpha_mwgl = [0.1,0.05,0.02,0.01,0.005,0];
    len_alpha_mwgl = length(alpha_mwgl);
    
    for i = 1:12
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = [alpha_mwgl(i)*N2,alpha_mwgl(i)*N1];
        param.pd_type = 'cartesian';
        param.inv_compute = 'eig';
        param.max_iter = 10000;
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = mwgl(X_M,param);
        graphs1_mwgl(:,i) = -L1(tril(true(N1),-1));
        graphs2_mwgl(:,i) = -L2(tril(true(N2),-1));
        
    end
    toc;
end

%% Save
filename = "molene_results.mat";
save(filename, "graphs1_mwgl", "graphs2_mwgl", "graphs1_teralasso", "graphs2_teralasso", "graphs1_rpgl", "graphs2_rpgl");

%% Visualize learned graphs
A2 = squareform(graphs2_mwgl(:,20));
A2(A2<1e-4) = 0;
G = graph(A2);
plot(G,'XData',x,'YData',y,'ZData',z);
% scatter3(x,y,z);