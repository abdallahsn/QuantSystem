from modules.meta_learner import MetaLearnerLSTM


def test_meta_learner_uses_runtime_meta_feature_dim():
    model = MetaLearnerLSTM(
        seq_len=50,
        n_stat_feat=31,
        n_meta_feat=6,
        n_visual_emb=8,
        brain_file='__nonexistent__.keras',
    )

    assert model.n_stat == 31
    assert model.n_meta == 6
    assert model.n_visual == 8
    assert model.n_total == 45

