from email.mime import audio
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os
import random
from modules.audio.core import load_audio
from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, audio_paths, transcripts, sos_id, eos_id, config, dataset_path, audio_extension, processor):
        super().__init__()

        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.data_size = len(self.audio_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.config = config
        self.dataset_path = dataset_path
        self.audio_extension = audio_extension
        self.processor = processor
        self.shuffle()
    
    def __getitem__(self, idx):
        audio = load_audio(os.path.join(self.dataset_path, self.audio_paths[idx]), del_silence=self.config.del_silence, extension=self.audio_extension)

        if audio is None:
            return None

        audio = torch.FloatTensor(audio)
        # audio = self.processor(audio, sampling_rate=16000).input_values[0]
        transcript, status = self.parse_transcript(self.transcripts[idx])

        if status == 'err':
            print(self.transcripts[idx])
            print(idx)
        return audio, transcript

    def parse_transcript(self, transcript):
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            try:
                transcript.append(int(token))
                status='nor'
            except Exception:
                print(tokens)
                status='err'
        transcript.append(int(self.eos_id))

        return transcript, status

    def shuffle(self):
        tmp = list(zip(self.audio_paths, self.transcripts))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts = zip(*tmp)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def count(self):
        return len(self.audio_paths)


def load_dataset(transcripts_path):

    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            try:
                audio_path, korean_transcript, transcript = line.split('\t')
            except Exception:
                print(line)
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


def split_dataset(config, transcripts_path, vocab, processor, valid_size=.2):
    print("split dataset start!")

    audio_paths, transcripts = load_dataset(transcripts_path)

    train_audio_paths, valid_audio_paths, train_transcripts, valid_transcripts = train_test_split(audio_paths,
                                                                                                transcripts,
                                                                                                test_size=valid_size)

    tmp = list(zip(train_audio_paths, train_transcripts))
    random.shuffle(tmp)
    train_audio_paths, train_transcripts = zip(*tmp)

    train_dataset = AudioDataset(train_audio_paths, train_transcripts, vocab.sos_id, vocab.eos_id, 
                                config=config, dataset_path=config.dataset_path, audio_extension=config.audio_extension, processor=processor)
    valid_dataset = AudioDataset(valid_audio_paths, valid_transcripts, vocab.sos_id, vocab.eos_id, 
                                config=config, dataset_path=config.dataset_path, audio_extension=config.audio_extension, processor=processor)

    return train_dataset, valid_dataset

def collate_fn(batch):

    pad_id = 0
    """ functions that pad to the maximum sequence length """

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    try:
        batch = [i for i in batch if i != None] 
        batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)


        seq_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) - 1 for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)


        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(pad_id)

        for x in range(batch_size):
            
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)

            seqs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        seq_lengths = torch.IntTensor(seq_lengths)
        return seqs, targets, seq_lengths, target_lengths

    except Exception as e:
        print(e)
