import torch
import torch.nn as nn


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))

    def _get_lstm_features(self, sentence):
        # sentence: (batch_size, max_seq_len)
        embeds = self.embedding(sentence)  # (batch_size, max_seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, max_seq_len, hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # (batch_size, max_seq_len, tagset_size)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # feats: (batch_size, max_seq_len, tagset_size)
        # tags: (batch_size, max_seq_len)
        batch_size = feats.size(0)
        score = torch.zeros(batch_size, device=feats.device)
        start_tag = torch.full((batch_size, 1), self.tagset_size - 1, dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tag, tags], dim=1)  # (batch_size, max_seq_len + 1)

        for i in range(feats.size(1)):  # 遍历序列长度
            feat = feats[:, i, :]  # (batch_size, tagset_size)
            score = score + self.transitions[tags[:, i], tags[:, i + 1]] + feat[range(batch_size), tags[:, i + 1]]
        return score

    def _forward_alg(self, feats):
        # feats: (batch_size, max_seq_len, tagset_size)
        batch_size = feats.size(0)
        init_alphas = torch.full((batch_size, self.tagset_size), -10000., device=feats.device)
        init_alphas[:, self.tagset_size - 1] = 0.  # START tag
        forward_var = init_alphas

        for i in range(feats.size(1)):
            feat = feats[:, i, :]  # (batch_size, tagset_size)
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[:, next_tag].unsqueeze(1).expand(batch_size, self.tagset_size)
                trans_score = self.transitions[next_tag, :self.tagset_size].unsqueeze(0).expand(batch_size,
                                                                                                self.tagset_size)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            forward_var = torch.stack(alphas_t, dim=1)
        return torch.logsumexp(forward_var, dim=1)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # (batch_size, max_seq_len, tagset_size)
        forward_score = self._forward_alg(feats)  # (batch_size,)
        gold_score = self._score_sentence(feats, tags)  # (batch_size,)
        return (forward_score - gold_score).mean()  # 平均损失

    def forward(self, sentence):
        feats = self._get_lstm_features(sentence)  # (batch_size, max_seq_len, tagset_size)
        score, tag_seq = self.viterbi_decode(feats)
        return score, tag_seq

    def viterbi_decode(self, feats):
        # feats: (batch_size, max_seq_len, tagset_size)
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        init_vvars = torch.full((batch_size, self.tagset_size), -10000., device=feats.device)
        init_vvars[:, self.tagset_size - 1] = 0  # START tag
        forward_var = init_vvars

        backpointers = []
        for i in range(seq_len):
            feat = feats[:, i, :]  # (batch_size, tagset_size)
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                trans = self.transitions[next_tag, :self.tagset_size].unsqueeze(0).expand(batch_size, self.tagset_size)
                next_tag_var = forward_var + trans
                best_tag_id = torch.argmax(next_tag_var, dim=1)  # (batch_size,)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[range(batch_size), best_tag_id])
            forward_var = torch.stack(viterbivars_t, dim=1) + feat
            backpointers.append(torch.stack(bptrs_t, dim=1))  # (batch_size, tagset_size)

        best_tag_id = torch.argmax(forward_var, dim=1)  # (batch_size,)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(batch_size), best_tag_id]
            best_path.append(best_tag_id)
        best_path.reverse()
        tag_seq = torch.stack(best_path[1:], dim=1)  # (batch_size, max_seq_len)

        # 计算最佳路径的分数
        score = torch.zeros(batch_size, device=feats.device)
        score += feat[range(batch_size), tag_seq[:, 0]]
        for i in range(1, seq_len):
            score += self.transitions[tag_seq[:, i - 1], tag_seq[:, i]]
            score += feats[:, i, :][range(batch_size), tag_seq[:, i]]

        return score, tag_seq