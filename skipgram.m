%% demo_skipgram_ver1.m
% Skip-Gram モデルの実装

clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. コーパスの定義
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


corpus = { ...
    'I like cats', ...
    'I love dogs', ...
    'You like cats', ...
    'They love animals' ...
    };

% corpus = { ...
%     "She walked to the store."
%     "He bought some fresh apples."
%     "They enjoyed the sunny weather."
%     "The cat sat on the windowsill."
%     "She read a fascinating book."
%     "He cooked dinner for the family."
%     "They watched a movie together."
%     "The dog chased the ball."
%     "She wrote a letter to her friend."
%     "He painted a beautiful picture."
%     "They went for a morning jog."
%     "The bird sang a lovely song."
%     "She planted flowers in the garden."
%     "He fixed the broken chair."
%     "They listened to their favorite music."
%     "The child played in the sandbox."
%     "She baked cookies for the neighbors."
%     "He built a treehouse in the backyard."
%     "They traveled to the mountains."
%     "The sun set behind the hills."
%     "She swam in the ocean."
%     "He like to play video games."
%     };

% corpus = {
%     % 吾輩は猫である
%     "I am a cat. I don’t yet have a name.
% I have no clue where I was born. All I remember is that I was crying “meow, meow” somewhere dark and damp. It was there that I first laid eyes on what they call human beings. Later I learned that he was a student—said to be the most savage variety among humans. They say these students sometimes catch creatures like us and boil them for food. But back then, I didn’t have a single thought in my head, so I wasn’t particularly scared. All I recall is a strange floating sensation when I was lifted in his palm. Once I’d settled down a bit on that palm, I took a good look at the student’s face; that must have been my very first glimpse of a human. I still remember how peculiar it felt. For one thing, a face that should’ve been covered with fur was instead completely smooth, much like a metal kettle. Since then, I’ve run across quite a few cats, but never one so oddly shaped. What’s more, the center of his face stuck out too far, and smoke would occasionally puff out with a “puh, puh” sound from the hole there. It made me choke and gave me quite a hard time. Only recently did I figure out that this was what humans call tobacco."
%     % 坊っちゃん
%     "I inherited my recklessness from my parents, and it’s brought me nothing but trouble ever since I was a kid. Back in elementary school, I once jumped off the second floor of the school building and ended up so paralyzed that I couldn’t walk for about a week. You might ask why I did something so rash. There wasn’t really any serious reason. I was sticking my head out of the second floor of a newly built wing when one of my classmates teased me, “No matter how tough you act, you’d never be able to jump from there, you coward!” So I jumped.
% When I came home on the janitor’s back, my father glared at me with wide eyes and said, “Who in their right mind jumps off a second floor and throws out their back?” So I answered, “Next time, I’ll jump without throwing it out.”"
% }




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. 語彙リストの作成，ID割当
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% トークン化
allTokens = {};
for i = 1:length(corpus)
    % 英語の場合，空白区切りによる分割が可能
    tokens = split(corpus{i}, ' ');
    allTokens = [allTokens; tokens];
end

% ユニーク単語リスト
uniqueWords = unique(allTokens);
vocabSize   = length(uniqueWords);

% 単語をIDにマッピング
word2id = containers.Map();
id2word = cell(vocabSize, 1);
for i = 1:vocabSize
    word2id(uniqueWords{i}) = i;
    id2word{i} = uniqueWords{i};
end

disp('語彙数：', vocabSize);
disp('文章数：', length(corpus));
disp('語彙リスト(ランダム20個)：', uniqueWords(randperm(vocabSize, 20)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Skip-Gram用データ (center, context) ペアの作成
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
windowSize = 2;  % 周辺単語をどのくらい見るか
trainingPairs = [];  % [center_word_id, context_word_id] のペアを格納

for i = 1:length(corpus)
    tokens = split(corpus{i}, ' ');
    % 単語ID列に変換
    tokenIDs = arrayfun(@(x) word2id(x{1}), tokens);  
    
    for j = 1:length(tokenIDs)
        centerID = tokenIDs(j);
        % 周辺 windowSize 分だけコンテキストを取得
        for k = max(1, j - windowSize) : min(length(tokenIDs), j + windowSize)
            if k ~= j
                contextID = tokenIDs(k);
                trainingPairs = [trainingPairs; [centerID, contextID]];
            end
        end
    end
end

disp('語彙，IDペア (ランダム10個)：', trainingPairs(randperm(size(trainingPairs, 1), 10), :));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. モデル構造定義 (2層)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
embeddingDim = 5;  % 埋め込みベクトルの次元（例：5次元）

% 重み初期化
% 入力→中間層（埋め込み層）: vocabSize x embeddingDim
W1 = 0.01 * randn(vocabSize, embeddingDim);
% 中間層→出力層 (embeddingDim x vocabSize)
W2 = 0.01 * randn(embeddingDim, vocabSize);

% 学習率
learningRate = 0.01;

% 反復回数（エポック）
numEpochs = 2000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. 学習ループ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:numEpochs
    % 学習データからランダムに1サンプル取って更新 (SGD)
    idx = randi(size(trainingPairs, 1));  % 乱数でペア選択
    centerID  = trainingPairs(idx, 1);
    contextID = trainingPairs(idx, 2);
    
    % ------- Forward pass -------
    % 入力（one-hot）
    x = zeros(vocabSize, 1);
    x(centerID) = 1;
    
    % 中間層（埋め込みベクトル） h = x^T * W1
    %   x は vocabSize x 1,  W1 は vocabSize x embeddingDim
    %   => h は 1 x embeddingDim
    h = (x' * W1)';
    
    % 出力層 u = W2^T * h
    %   W2 は embeddingDim x vocabSize
    %   => u は vocabSize x 1
    u = W2' * h;
    
    % softmax（確率変換）
    y_pred = softmax(u);
    
    % ------- Loss (交差エントロピー) -------
    % 正解は contextID が 1 の one-hotベクトル
    t = zeros(vocabSize, 1);
    t(contextID) = 1;
    
    % 損失関数 L = -sum(t .* log(y_pred))
    L = -sum(t .* log(y_pred + 1e-15)); % 零割防止
    
    % ------- Backward pass -------
    % 出力層での勾配 dU = (y_pred - t)
    % => dU は vocabSize x 1
    dU = (y_pred - t);
    
    % W2 の勾配 dW2
    % h は embeddingDim * 1, dU は vocabSize * 1
    % => dW2 は embeddingDim * vocabSize
    dW2 = h * dU';
    
    % 中間層（h）への勾配 dh = W2 * dU
    % W2 は embeddingDim * vocabSize, dU は vocabSize * 1 
    % => dh は embeddingDim * 1
    dh = W2 * dU;
    
    % W1 の勾配 dW1
    % x は vocabSize * 1, dh は embeddingDim * 1
    % => dW1 は vocabSize * embeddingDim
    % ただし中心単語 centerID だけ勾配が入る（one-hot）
    dW1 = zeros(size(W1));
    dW1(centerID, :) = dh';
    
    % ------- パラメータ更新 (SGD) -------
    W2 = W2 - learningRate * dW2;
    W1 = W1 - learningRate * dW1;
    
    % ------- 学習過程の可視化      -------
    if mod(epoch, 500) == 0
        fprintf('Epoch %d / %d\n', epoch, numEpochs);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. 学習後の単語ベクトルを確認
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W1 が単語ごとの埋め込みベクトルとして機能 (vocabSize x embeddingDim)
embeddings = W1;

disp('単語 -> 埋め込みベクトル(一部)');
for i = 1:vocabSize
    fprintf('%s: [', id2word{i});
    fprintf('%.3f ', embeddings(i, :));
    fprintf(']\n');
end

% 例) 単語 "like" と他の単語の類似度を算出
wordName = "like";
catsID = word2id(wordName);
catsVec = embeddings(catsID, :);

for i = 1:vocabSize
    otherVec = embeddings(i, :);
    sim = cosine_similarity(catsVec, otherVec);
    fprintf('%s - %s: %.3f\n', wordName, id2word{i}, sim);
end

%--------------------------------------------------
% 学習後の埋め込みベクトル（W1）と単語リスト
%--------------------------------------------------
embeddings = W1;   % vocabSize * embeddingDim
vocabSize  = size(embeddings, 1);  % 単語数
% id2word = {...};  % {1} = '単語1', {2} = '単語2', ...

%--------------------------------------------------
% 1. PCA を用いて 3 次元に次元圧縮 [Statistics and Machine Learning Toolbox]
%--------------------------------------------------
[coeff, score, ~, ~, explained] = pca(embeddings);
% score は単語数 * 主成分数 (ここでの行数は単語数)
% 上位3つの主成分を利用
X3D_pca = score(:, 1:3);

figure('Name','PCA 3D','NumberTitle','off');
scatter3(X3D_pca(:,1), X3D_pca(:,2), X3D_pca(:,3), 'filled');
title('Word Embeddings (PCA 3D)');
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
axis equal; grid on; hold on;

% 単語ラベルを表示
for i = 1:vocabSize
    text(X3D_pca(i,1), X3D_pca(i,2), X3D_pca(i,3), ...
        [' ' id2word{i}], ...
        'VerticalAlignment','bottom', ...
        'HorizontalAlignment','left');
end
hold off;

% PCA の寄与率を表示
disp('PCA 寄与率 (上位3主成分):');
disp(explained(1:3));

%--------------------------------------------------
% 2. T-SNE を用いて 3 次元に次元圧縮 [Statistics and Machine Learning Toolbox]
%--------------------------------------------------
perplexityValue = 5;
X3D_tsne = tsne(embeddings, ...
    'NumDimensions', 3, ...
    'Perplexity', perplexityValue);

figure('Name','t-SNE 3D','NumberTitle','off');
scatter3(X3D_tsne(:,1), X3D_tsne(:,2), X3D_tsne(:,3), 'filled');
title('Word Embeddings (t-SNE 3D)');
xlabel('Dimension 1'); ylabel('Dimension 2'); zlabel('Dimension 3');
axis equal; grid on; hold on;

% 単語ラベルを表示
for i = 1:vocabSize
    text(X3D_tsne(i,1), X3D_tsne(i,2), X3D_tsne(i,3), ...
        [' ' id2word{i}], ...
        'VerticalAlignment','bottom', ...
        'HorizontalAlignment','left');
end
hold off;

