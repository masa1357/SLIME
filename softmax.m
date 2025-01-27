function y = softmax(x)
    % x はベクトル (vocabSize x 1 など)
    ex = exp(x - max(x));  % オーバーフロー防止のため最大値を引く
    y = ex / sum(ex);
end
