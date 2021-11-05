import os

folder_path = INSERT_YOUR_DATASET_PATH_HERE # Anonymized for submission.
files = [("en_es-fr.es", "en_es-fr.fr"),
         ("fr_en-es.en", "fr_en-es.es"),
         ("es_en-fr.en", "es_en-fr.fr")
        ]

def dump_file(filename_pair, splits, split_name):
    open(os.path.join(folder_path, filename_pair[0] + '.' + split_name), 'w+').writelines(splits[0])
    open(os.path.join(folder_path, filename_pair[1] + '.' + split_name), 'w+').writelines(splits[1])

for file_pair in files:
    source_path = os.path.join(folder_path, file_pair[0])
    target_path = os.path.join(folder_path, file_pair[1])
    source_sentence = open(source_path).readlines()
    target_sentence = open(target_path).readlines()
    
    assert len(source_sentence) == len(target_sentence)
    assert len(source_sentence) > 4000
    
    test_splits  = (source_sentence[-1000:], target_sentence[-1000:])
    dev_splits   = (source_sentence[-2000:-1000], target_sentence[-2000:-1000])
    train_splits = (source_sentence[:-2000], target_sentence[:-2000])

    dump_file(file_pair, train_splits, 'train')
    dump_file(file_pair, dev_splits, 'dev')
    dump_file(file_pair, test_splits, 'test')
    
