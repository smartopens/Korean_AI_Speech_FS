import re
import os
import pandas as pd


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")
    drop_index = set()

    # freq 30 이하
    normal_labels = {"쥔", "옜", "썻", "훗", "떳", "휠", "퀵", "쐈", "끙", "솟", "맷", "쉼", "샾", "챈", "쏜", "낱", "슉",
                     "꾀", "톰", "꺄", "뗀", "쯧", "윷", "좆", "텁", "듦", "캬", "얍", "몹", "댐", "멱", "샥", "킁", "쌋", "팻",
                     "꺽", "튄", "짊", "콰", "힝", "덫", "퉤", "쉿"}


    # freq 30 ~ 100
    drop_labels = {"밲", "폿", "죙", "띡", "홋", "눴"}
    substition_labels = {"옇": "였", "쥴": "줄", "됬": "됐"}

    for i in ch_labels.index:
        if ch_labels['freq'][i] <= 30:

            if ch_labels['char'][i].isdigit() or ch_labels['char'][i] in normal_labels:
                continue

            drop_index.add(i)

    print("!!!s")


    ch_labels = ch_labels.drop(drop_index)
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]


def generate_character_script(data_df, labels_dest):
    print('[INFO] create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, "labels.csv"))
    cnt = 0

    with open(os.path.join(labels_dest,"transcripts.txt"), "w+") as f:
        for audio_path, transcript in data_df.values:
            char_id_transcript = sentence_to_target(transcript, char2id)
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')
            if cnt >= 50:
                continue
            print(audio_path, transcript, char_id_transcript)
            cnt += 1


def preprocessing(transcripts_dest, labels_dest):
    transcript_df = pd.read_csv(transcripts_dest)
    generate_character_script(transcript_df, labels_dest)

    print('[INFO] Preprocessing is Done')