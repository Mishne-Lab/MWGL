function GS = nti(S,param)

max_err = param.max_err;
delta_err = param.delta_err;

N = size(S,1);
p = N*(N-1)/2;

[V,D] = eig(S);
[~,ind] = sort(diag(D));
V = V(:,ind);

switch param.gso
    case 'adjacency'
        switch param.template
            case 'noiseless'
                cvx_begin quiet
                variable GS(N,N) symmetric
                variable lambda(N)
                    minimize(norm(GS(:),1))
                    subject to
                    GS==V*diag(lambda)*V';
                    GS(:)>=0
                    abs(diag(GS))<=1e-6
                    GS*ones(N,1)>=1
                cvx_end
            case 'noisy'
                while true
                    cvx_begin quiet
                    variable GS(N,N) symmetric
                    variable lambda(N)
                        minimize(norm(GS(:),1))
                        subject to
                        GS(:)>=0
                        abs(diag(GS))<=1e-6
                        GS*ones(N,1)>=1
                        norm(GS-V*diag(lambda)*V','fro')<=max_err
                    cvx_end
                    if cvx_status == "Solved"
                        break
                    end
                    max_err = max_err+delta_err;
                end
            case 'incomplete'
        end
    case 'laplacian'
        switch param.template
            case 'noiseless'
            case 'noisy'
                B = zeros(N-3,N);
                for i = 1:N-3
                    B(i,i) = 1;
                    B(i,i+3) = -1;
                end
                while true
                    cvx_begin quiet
                    variable GS(N,N) symmetric
                    variable lambda(N)
                        minimize(norm(GS(:),1))
                        subject to
                        GS(tril(true(N),-1))<=0
                        GS*ones(N,1)==0
                        B*lambda>=0.1
                        norm(GS-V*diag(lambda)*V','fro')<=max_err
                    cvx_end
                    if cvx_status == "Solved"
                        break
                    end
                    max_err = max_err+delta_err;
                end
            case 'incomplete'
        end
end

end