function DI = dataToimage_S(data, ps, ss, m, n)
% 基于变化量相似性的加权映射，以空间最近的块为参考（权重1）
% 输入：
%   data   - 每个块的变化量，列向量，长度 N
%   ps, ss - 块半径和步长
%   m, n   - 原始图像尺寸
% 输出：
%   DI     - m×n 差异图

    data = data(:);
    Nblocks = length(data);

    % 填充后的尺寸
    H = m + 2*ps;
    W = n + 2*ps;

    % 生成块中心网格（与 imageTodata 一致）
    row_centers = (1+ps):ss:(H-ps);
    if row_centers(end) ~= (H-ps)
        row_centers(end+1) = H-ps;
    end
    col_centers = (1+ps):ss:(W-ps);
    if col_centers(end) ~= (W-ps)
        col_centers(end+1) = W-ps;
    end
    R = length(row_centers);
    C = length(col_centers);
    N = R * C;
    if N ~= Nblocks
        error('data长度(%d)与块总数(%d)不一致', Nblocks, N);
    end

    % 记录每个块的中心坐标（填充后坐标系）
    block_centers = zeros(N, 2);
    idx = 0;
    for r = 1:R
        for c = 1:C
            idx = idx + 1;
            block_centers(idx, :) = [row_centers(r), col_centers(c)];
        end
    end

    % 构建从坐标到索引的映射（用于快速查找）
    center_to_idx = zeros(H, W, 'int32');
    for idx = 1:N
        cx = block_centers(idx, 1);
        cy = block_centers(idx, 2);
        center_to_idx(cx, cy) = idx;
    end

    % 初始化填充后的差异图
    DI_padded = zeros(H, W);

    % 遍历原始图像区域内的每个像素
    for i = 1+ps : H-ps
        for j = 1+ps : W-ps
            % 找到覆盖当前像素的所有块索引
            rows = local_centers(i, ps, ss, H);
            cols = local_centers(j, ps, ss, W);
            if isempty(rows) || isempty(cols)
                continue;
            end
            block_idxs = [];
            for ri = 1:length(rows)
                cx = rows(ri);
                for cj = 1:length(cols)
                    cy = cols(cj);
                    bidx = center_to_idx(cx, cy);
                    if bidx > 0
                        block_idxs(end+1) = bidx;
                    end
                end
            end
            if isempty(block_idxs)
                continue;
            end

            % 在这些覆盖块中，找到空间上距离(i,j)最近的块中心
            min_dist2 = inf;
            ref_idx = block_idxs(1);
            for k = 1:length(block_idxs)
                bidx = block_idxs(k);
                dx = i - block_centers(bidx, 1);
                dy = j - block_centers(bidx, 2);
                dist2 = dx*dx + dy*dy;
                if dist2 < min_dist2
                    min_dist2 = dist2;
                    ref_idx = bidx;
                end
            end

            % 参考块的变化量
            v_ref = data(ref_idx);

            % 计算每个覆盖块的权重：基于与v_ref的变化量差异
            vals = data(block_idxs);
            % 计算局部标准差（用于自适应宽度）
            sigma = std(vals);
            if sigma < 1e-6
                % 所有变化量几乎相等，直接平均
                DI_padded(i,j) = mean(vals);
                continue;
            end
            % 权重：与v_ref差异越小，权重越大
            weights = exp( - (vals - v_ref).^2 / sigma );
            % 确保参考块自身的权重为1（可能因数值误差略小于1，强行归一化）
%             max_w = max(weights);
%             if max_w > 0
%                 weights = weights / max_w;
%             end
            DI_padded(i,j) = sum(weights .* vals) / sum(weights);
        end
    end

    % 裁剪出原始图像区域
    DI = DI_padded(ps+1:ps+m, ps+1:ps+n);
end

% 辅助函数：获取覆盖位置 pos 的所有块中心坐标（填充后坐标系）
function centers = local_centers(pos, ps, ss, H)
    start = 1 + ps;
    step = ss;
    last = H - ps;
    low = max(start, pos - ps);
    high = min(last, pos + ps);
    if low > high
        centers = [];
        return;
    end
    first = ceil((low - start) / step) * step + start;
    if first > high
        centers = [];
        return;
    end
    centers = first:step:high;
end