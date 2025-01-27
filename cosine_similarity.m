
function sim = cosine_similarity(vec1, vec2)
    sim = (vec1(:)' * vec2(:)) / (norm(vec1) * norm(vec2));
end
