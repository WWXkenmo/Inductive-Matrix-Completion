# Inductive-Matrix-Completion
%we proposed a type of inductive matrix completion with trace norm regularizer, and presented two practical method for solving trace norm based IMC optimization problem. Using two different tasks and four different benchmark tasks, We showed the max-norm can often be superior to established trace-norm regularization. 

%Here, we present the case 'drug-protein prediction' with max-norm based IMC, 10-fold validation test
%with sampling ratio 1:2 (positive samples size:negative samples size)
%using addpath to import direction of codes and datasets
clear all
%% loding dataset
P = importdata('mat_drug_protein.txt');
% find positive and negative interactions
Pint = find(P); % pair of P
Nint = length(Pint);
Pnoint = find(~P);
Pnoint = Pnoint(randperm(length(Pnoint), floor(Nint * 2)));
Nnoint = length(Pnoint);
posFilt = crossvalind('Kfold', Nint, 10);
negFilt = crossvalind('Kfold', Nnoint, 10);

% load feature
X = importdata('drug_vector_d100.txt');
Y = importdata('protein_vector_d400.txt');
d_emb1=size(X,2);
d_emb2=size(Y,2);
d=size(X,1);
D=size(Y,1);
AUROC_IMC = zeros(10, 1);
AURPC_IMC = zeros(10, 1);

%set parameters
latent_dims=1;
t=1;
miu=5;
gamma=0.5;
alpha=0.5; 
epsilon=10^-3;
AUROC = zeros(10, 1);
AURPC = zeros(10, 1);
bs=1;
for foldID = 1 : 10
		train_posIdx = Pint(posFilt ~= foldID,:);
		train_negIdx = Pnoint(negFilt ~= foldID,:);
		train_idx = [train_posIdx; train_negIdx];
		Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];
		fprintf('Train data: %d positives, %d negatives\n', sum(Ytrain == 1), sum(Ytrain == 0));
		test_posIdx = Pint(posFilt == foldID,:);
		test_negIdx = Pnoint(negFilt == foldID,:);
		test_idx = [test_posIdx; test_negIdx];
		Ytest = [ones(length(test_posIdx), 1); zeros(length(test_negIdx), 1)];		
		fprintf('Test data: %d positives, %d negatives\n', sum(Ytest == 1), sum(Ytest == 0));
        
		%reprocessing
		[I, J] = ind2sub(size(P), train_idx);
		train = [I,J];
		train_posIdx = train(find(Ytrain==1),:);
		train_negIdx = train(find(Ytrain==0),:);
		[I, J] = ind2sub(size(P), test_idx);
		test = [I,J];
		test_posIdx = test(find(Ytest==1),:);
		test_negIdx = test(find(Ytest==0),:);
        
		
		[score,Z,L,R] = IMCmaxNorm(X,Y, P, train_posIdx, train_negIdx,latent_dims, t, miu, epsilon, gamma, alpha,1);
		[trainroc, trainpr] = auc2(Ytrain, score(train_idx), 0);
		[testroc, testpr] = auc2(Ytest, score(test_idx), 0);
		
		AUROC(foldID) = testroc;
		AURPC(foldID) = testpr;
		fprintf('Fold %d, Train: AUROC=%f AUPR=%f; Test: AUROC=%f, AUPR=%f\n', foldID, trainroc, trainpr, testroc, testpr);
end
