%% Load molene and learned graphs
load("meteo_molene_t.mat");
X = value;
x = info{4}; y = info{3}; z = info{5}; coords = [x,y,5*z];
N2 = size(X,1);

%% Plot learned Laplacians
figure(1)
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
    if i == 1
        ylabel('PGL','FontSize',24)
    end
end
for i = 10:-1:6
    nexttile
    pm = reshape(graphs2_teralasso(:,1*(i-1)+1,1*(i-1)+1),N2,N2);
    imagesc(-pm+diag(diag(pm)));
    c=colorbarpzn(min(pm(:)), -min(pm(:)),'rev');c.FontSize=16;
    axis square
    xticks([]);yticks([]);
    if i == 10
        ylabel('TeraLasso','FontSize',24)
    end
end
for i = 1:5
    nexttile
    w = graphs2_mwgl(:,1*(i-1)+1);
    imagesc(squareform(w));
    c=colorbarpzn(0, max(w),'rev');c.FontSize=16;
    axis square
    xticks([]);yticks([]);
    if i == 1
        ylabel('MWGL (ours)','FontSize',24)
    end
end
% print('molene_laplacian_compare','-dpng','-r600')