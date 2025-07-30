clc; clear; close all;
% 개선코드별 안내멘트 정의
code2message = containers.Map('KeyType','double','ValueType','char');
code2message(0) = "이 계획은 우선순위가 낮게 설정되어 있습니다. 우선순위를 높이는 것을 고려해보세요.";
code2message(1) = "목표시간이 실제 소요시간보다 많이 짧습니다. 조금 더 여유를 두는 것이 좋습니다.";
code2message(2) = "목표시간이 실제 소요시간보다 많이 깁니다. 시간을 적절히 줄여보세요.";
code2message(3) = "전체 계획의 총 목표시간이 많이 깁니다. 전체 시간을 조절해보세요.";
code2message(4) = "계획 항목 수가 너무 많습니다. 일부 항목을 정리하거나 나중으로 미루는 것도 좋은 전략입니다.";
% 파일 불러오기
trainFolder = "C:\final_plan\plan";
newPlanFile = "C:\final_plan\new_plan.csv";
fileList = dir(fullfile(trainFolder, "plan_*.csv"));
trainT = table();
for i = 1:length(fileList)
   fpath = fullfile(trainFolder, fileList(i).name);
   T = readtable(fpath, 'VariableNamingRule','preserve');
   T.Category = string(T.Category);
   T.SubCategory = string(T.SubCategory);
   trainT = [trainT; T];
end
newPlan = readtable(newPlanFile, 'VariableNamingRule','preserve');
trainT.Properties.VariableNames = {'Category','SubCategory','Priority','TargetTime','ActualTime','TaskSuccess','PlanSuccess'};
newPlan.Properties.VariableNames = {'Category','SubCategory','Priority','TargetTime'};
trainT.Category = string(trainT.Category);
trainT.SubCategory = string(trainT.SubCategory);
newPlan.Category = string(newPlan.Category);
newPlan.SubCategory = string(newPlan.SubCategory);
newPlan.SubCategory(ismissing(newPlan.SubCategory)) = "0";
trainT.SubCategory(ismissing(trainT.SubCategory)) = "0";
allCats = unique([trainT.Category; newPlan.Category]);
allSubs = unique([trainT.SubCategory; newPlan.SubCategory]);
[~, trainT.CategoryNum] = ismember(trainT.Category, allCats);
[~, trainT.SubCategoryNum] = ismember(trainT.SubCategory, allSubs);
[~, newPlan.CategoryNum] = ismember(newPlan.Category, allCats);
[~, newPlan.SubCategoryNum] = ismember(newPlan.SubCategory, allSubs);
trainT.Priority = double(trainT.Priority);
trainT.TargetTime = double(trainT.TargetTime);
trainT.PlanSuccess = double(trainT.PlanSuccess);
newPlan.Priority = double(newPlan.Priority);
newPlan.TargetTime = double(newPlan.TargetTime);
%% 개선 코드 라벨 초기화
trainT.ImprovementCodes = repmat({[]}, height(trainT), 1);
for i = 1:height(trainT)
   codes = [];
   if trainT.TaskSuccess(i) == 1 && abs(trainT.TargetTime(i) - trainT.ActualTime(i)) <= 30
       trainT.ImprovementCodes{i} = [];
       continue;
   end
   if trainT.TargetTime(i) < trainT.ActualTime(i) - 60
       codes(end+1) = 1;  % 목표시간 늘리기
   end
   if trainT.TargetTime(i) > trainT.ActualTime(i) + 60
       codes(end+1) = 2;  % 목표시간 줄이기
   end
   if trainT.Priority(i) >= 2
       codes(end+1) = 0;  % 우선순위 높이기
   end
   trainT.ImprovementCodes{i} = unique(codes);
end
%% 모델 학습
X_train = [trainT.CategoryNum, trainT.SubCategoryNum, trainT.Priority, trainT.TargetTime];
Y_train = trainT.PlanSuccess;
classificationModel = fitctree(X_train, Y_train, 'CategoricalPredictors', [1 2]);
seqX = num2cell(X_train', [1]);
layers = [
   sequenceInputLayer(4)
   lstmLayer(10, 'OutputMode','last')
   fullyConnectedLayer(1)
   sigmoidLayer
   regressionLayer];
options = trainingOptions('adam','MaxEpochs',30,'MiniBatchSize',8,'Verbose',false);
net = trainNetwork(seqX, Y_train, layers, options);
%% 예측 및 개선 분석
successProbs = zeros(height(newPlan),1);
improvements = strings(height(newPlan),1);
improvementsNumbered = strings(height(newPlan),1);
avgPriority = mean(trainT.Priority(trainT.PlanSuccess == 1));
avgTargetTime = mean(trainT.TargetTime(trainT.PlanSuccess == 1));
for i = 1:height(newPlan)
   x = [newPlan.CategoryNum(i), newPlan.SubCategoryNum(i), newPlan.Priority(i), newPlan.TargetTime(i)];
   % 예측
   [~, treeScore] = predict(classificationModel, x);
   lstmScore = predict(net, num2cell(x', [1]))';
   prob = 0.7 * treeScore(2) + 0.3 * lstmScore;
   successProbs(i) = prob;
   % 개선 제안
   imp = strings(0);
  
   % 예측값 기반 개선
   if prob <= 0.6
       if newPlan.TargetTime(i) > avgTargetTime + 30
           imp(end+1) = code2message(2);  % 목표시간 줄이기
       elseif newPlan.TargetTime(i) < avgTargetTime - 30
           imp(end+1) = code2message(1);  % 목표시간 늘리기
       end
       if newPlan.Priority(i) > 1 && newPlan.Priority(i) < avgPriority
           imp(end+1) = code2message(0);  % 우선순위 올리기
       end
   end
  
   % 총 목표시간 24시간 초과 시 안내멘트 출력
   if newPlan.TargetTime(i) > 1440
       imp(end+1) = code2message(3);  % 총 목표시간 초과
   end
  
   % 유사 케이스 기반 개선코드 수집
   dists = sum((X_train - x).^2, 2);
   [~, idx] = sort(dists);
   topK = 5;
   neighborCodes = [];
   for j = 1:topK
       cid = trainT.ImprovementCodes{idx(j)};
       if ~isempty(cid)
           neighborCodes = [neighborCodes, cid];
       end
   end
   if ~isempty(neighborCodes)
       u = unique(neighborCodes);
       for k = 1:length(u)
           if isKey(code2message, u(k))
               imp(end+1) = "유사 계획 추천: " + code2message(u(k));
           end
       end
   end
  
   % 최종 저장
   improvements(i) = ifelse(isempty(imp), "-", strjoin(imp, newline));
   improvementsNumbered(i) = sprintf("%d번 항목: 성공확률 = %.1f%%\n%s", i, successProbs(i)*100, improvements(i));
end
%% 전체 성공률(평균)
overallSuccessProb = mean(successProbs);
%% 출력(콘솔)
fprintf("\n 예측 및 개선 결과:\n");
for i = 1:height(newPlan)
   fprintf("%s\n\n", improvementsNumbered(i));
end
%% 시각화
figure('Name','개인 계획 성공확률 분석','NumberTitle','off','Color','w');
tiledlayout(1,2, 'Padding','compact', 'TileSpacing','compact');
% 텍스트 출력
nexttile;
text(0, 1.0, sprintf('전체 계획 성공확률 = %.1f%%', overallSuccessProb*100), 'FontSize', 13, 'FontWeight', 'bold');
yPos = 0.95;
spacing = 0.08;
for i = 1:height(newPlan)
   % 성공확률
   text(0, yPos, sprintf('%d번 항목: 성공확률 = %.1f%%', i, successProbs(i)*100), ...
       'FontSize', 10, 'Interpreter','none');
   yPos = yPos - 0.035;
   % 개선 멘트
   lines = splitlines(improvements(i));
   for j = 1:length(lines)
       text(0, yPos, lines{j}, 'FontSize', 9, 'Interpreter','none');
       yPos = yPos - 0.03;
   end
   yPos = yPos - 0.03;
end
axis off;
% 도넛 차트
nexttile;
theta = linspace(0,2*pi,100);
fill(cos(theta), sin(theta), [0.85 0.85 0.85]); hold on;
fill([0 cos(linspace(0,2*pi*overallSuccessProb,100))], [0 sin(linspace(0,2*pi*overallSuccessProb,100))], [0.2 0.6 1.0]);
fill(0.6*cos(theta), 0.6*sin(theta), 'w');
text(0, 0, sprintf('%.0f%%', overallSuccessProb*100), 'HorizontalAlignment', 'center', 'FontSize', 18, 'FontWeight', 'bold');
title('전체 성공확률');
axis equal off;
%% 예측 정확도 + LSTM 회귀오차 출력
X = [trainT.CategoryNum, trainT.SubCategoryNum, trainT.Priority, trainT.TargetTime];
Y_true = trainT.PlanSuccess;
% 예측값 저장
Y_tree = zeros(height(trainT),1);
Y_lstm = zeros(height(trainT),1);
Y_hybrid = zeros(height(trainT),1);
for i = 1:height(trainT)
   x = X(i,:);
   [~, scoreTree] = predict(classificationModel, x);
   predLSTM = predict(net, num2cell(x', [1]))';
   Y_tree(i) = scoreTree(2);          % Tree 확률
   Y_lstm(i) = predLSTM;              % LSTM 확률
   Y_hybrid(i) = 0.7 * scoreTree(2) + 0.3 * predLSTM;  % 앙상블
end
% 이진 분류 정확도 (threshold = 0.5)
Y_pred = Y_hybrid > 0.5;
accuracy = sum(Y_pred == Y_true) / length(Y_true);
% LSTM 회귀 오차 (MSE)
mse_lstm = mean((Y_lstm - Y_true).^2);
% 출력
fprintf("\n 예측 모델 성능\n");
fprintf("예측 정확도 (앙상블): %.2f%%\n", accuracy * 100);
fprintf("LSTM 회귀 MSE: %.4f\n", mse_lstm);
%% 보조함수
function out = ifelse(cond, tval, fval)
   if cond
       out = tval;
   else
       out = fval;
   end
end
