from pathlib import Path

hubert_root=Path('/projects/0/prjs0893/speech-training/scripts/pretraining/hubert/')
kmeans_models = hubert_root / 'kmeans_models'
kmeans_model_filename = kmeans_models / 'km_train_mfcc-model_100clusters'

cluster_check = hubert_root / 'cluster_check'
kmeans_labels = cluster_check / 'kmeans_labels'

speech_training = Path('/projects/0/prjs0893/speech-training')

pretrain_datasets = speech_training / 'data/pretraining-datasets/'
data = speech_training / 'data'

cgn_news_sentences = data / 'cgn_sentences/split_files'

common_voice = pretrain_datasets / 'cv_wav'

mls = pretrain_datasets / 'MLS_dutch_wav'

pretrain_manifest = data / 'manifests/dataset-960h-nl/train.tsv'


metadata = Path('/projects/0/prjs0893/speech-training/metadata')
cgn_news_phonemes = metadata / 'news_books_phonemes_zs.tsv'
mls_phonemes = metadata / 'dutch_mls_phonemes_zs.tsv'
cv_phonemes = metadata / 'dutch_cv_phonemes_zs.tsv'
