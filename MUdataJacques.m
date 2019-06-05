clear all
close all

filename = ['D:\UBC - Postdoc\Sensorimotor\Jacques Gaussian Mixture\MU_distribution_subject.xlsx'];
%data = xlsread(filename,2,'B1:C74');
dataFull = xlsread(filename,1,'B1:D74');

data = dataFull(:,1:2);

clear numM
dataNew = data*2;
x = 1:12;
y = 1:7;
for ii = 1:12
    for jj = 0:6
        numM(ii,jj+1) = length(find(sum([dataNew(:,1)==ii dataNew(:,2)==jj],2)==2));
    end
end
% numM = [numM; zeros(1,7)];
figure;
surf(numM);
view(0,90);
shading interp;
caxis([0 30]);
colorbar('location','northoutside');

dataSubject = {};
currSub = 1;
subjectData = [];
for i=1:length( dataFull )
    if dataFull(i,3) == currSub
        subjectData = [subjectData; 2*dataFull(i,1:2)];
    else
        dataSubject = [dataSubject, subjectData];
        currSub = dataFull(i,3);
        subjectData = dataFull(i,1:2);
    end
end

GMModels = cell(2,1); 
options = statset('MaxIter',100000,'TolX', 1.0000e-09);
rng('shuffle');
for j = 2:2
    GMModels{j} = fitgmdist_lmm(dataSubject,j,'options',options);
    %GMModels{j} = fitgmdist(dataNew,j,'options',options);
    fprintf('\n GM Mean for %i Component(s)\n',j)
    Mu = GMModels{j}.mu
    Sigma = GMModels{j}.Sigma
    AIC = GMModels{j}.AIC
end


figure;
figInd = [1 3];
for j = 1:2
    subplot(2,2,figInd(j))
    h1 = surf(numM);
    view(0,90);
    shading interp;
    caxis([0 30]);
    colorbar('off');
    h = gca;
    hold on
    ylim([1 12]);
    xlim([1 7]);
    title('Data');
    ylabel('Centroid location');
    xlabel('Distribution');
    subplot(2,2,figInd(j)+1)
    ezcontour(@(x1,x2)pdf(GMModels{j},[x2 x1]),[1 7 1 12])
    title(sprintf('GM Model - %i Component(s)',j));
    ylabel('Centroid location');
    xlabel('Distribution');
end

