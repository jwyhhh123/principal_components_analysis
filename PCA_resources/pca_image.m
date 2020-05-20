%% Load Dataset
opts = delimitedTextImportOptions("NumVariables", 19);

opts.DataLines = [1, Inf];
opts.Delimiter = " ";

opts.VariableNames = ["Class", "region-centroid-col", "region-centroid-row", "short-line-density-5", "short-line-density-2", "vedge-mean", "vegde-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

opts = setvaropts(opts, "Class", "EmptyFieldRule", "auto");

% Dataset with class names
Dataset = readtable("/Users/dayueyue/Desktop/Year2/MathModeling&decisionMaking/image-seg/Dataset.txt", opts);

% Normalize the whole dataset.
Dataset{:, 2:end} = normalize(Dataset{:, 2:end});
%%
clear opts
%% Datasets without class names
temp = Dataset(:,2:end);
TF = (Dataset.Class == 'CEMENT');
cement = Dataset(TF,:);
cement = cement(:,2:end);

TF = (Dataset.Class == 'BRICKFACE');
brickface = Dataset(TF,:);
brickface = brickface(:,2:end);

TF = (Dataset.Class == 'FOLIAGE');
foliage = Dataset(TF,:);
foliage = foliage(:,2:end);

TF = (Dataset.Class == 'SKY');
sky = Dataset(TF,:);
sky = sky(:,2:end);

TF = (Dataset.Class == 'PATH');
path = Dataset(TF,:);
path = path(:,2:end);

TF = (Dataset.Class == 'WINDOW');
window = Dataset(TF,:);
window = window(:,2:end);

TF = (Dataset.Class == 'GRASS');
grass = Dataset(TF,:);
grass = grass(:,2:end);

%% PCA
% matrix of 18 attributes
data = table2array(temp);
data1 = table2array(cement);
data2 = table2array(brickface);
data3 = table2array(foliage);
data4 = table2array(sky);
data5 = table2array(path);
data6 = table2array(window);
data7 = table2array(grass);

% Cross varidation (train: 80%, test: 20%)
cv = cvpartition(size(data,1),'HoldOut',0.2);
idx = cv.test;
rng('default');

% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

normdata = dataTrain;
normdatat = dataTest;

% no use, just assigning
normdata1 = data1;
normdata2 = data2;
normdata3 = data3;
normdata4 = data4;
normdata5 = data5;
normdata6 = data6;
normdata7 = data7;

% apply pca
[coeff,score,latent] = pca(normdata);
[coefft,scoret,latentt] = pca(normdatat);
latent = sort(latent,'descend')

% Calculate the covariance matrix
covarianceMatrix = cov(normdata)

% V: eigenvector; D: eigenvalue
[V,D] = eig(covarianceMatrix);

coeff

% plot the eigenvalues
figure(1)
plot(latent);
ax = gca;
ax.YAxis.Exponent = 2;
set(gca,'XTick',[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18])

% the first two PCs
z = normdata * coeff(:,1:3);
zt = normdatat * coefft(:,1:3);

z1 = normdata1 * coeff(:,1:3);
z2 = normdata2 * coeff(:,1:3);
z3 = normdata3 * coeff(:,1:3);
z4 = normdata4 * coeff(:,1:3);
z5 = normdata5 * coeff(:,1:3);
z6 = normdata6 * coeff(:,1:3);
z7 = normdata7 * coeff(:,1:3);

figure(2)
scatter3(z1(:,1),z1(:,2),z1(:,3),5,'r');
hold on;
scatter3(z2(:,1),z2(:,2),z2(:,3),5,'g');
hold on;
scatter3(z3(:,1),z3(:,2),z3(:,3),5,'b');
hold on;
scatter3(z4(:,1),z4(:,2),z4(:,3),5,'y');
hold on;
scatter3(z5(:,1),z5(:,2),z5(:,3),5,'m');
hold on;
scatter3(z6(:,1),z6(:,2),z6(:,3),5,'c');
hold on;
scatter3(z7(:,1),z7(:,2),z7(:,3),5,'k');

%scatter3(z(:,1),z(:,2),z(:,3),1.5);
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

%scatter(zt(:,1),zt(:,2),1.5);
%yticks([-2.50,-2.40,-2.30,-2.20,-2.10,-2.00,-1.90,-1.80,-1.70,-1.60,-1.50,-1.40,-1.30,-1.20,-1.10,-1.00,-0.90,-0.80,-0.70,-0.60,-0.50,-0.40,0]);