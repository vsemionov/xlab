# Copyright 2025 Victor Semionov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class SequenceDataset(BaseDataset):
    parent: TokenDataset

    def __init__(
            self,
            parent: TokenDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            concatenate: bool = False, pad_incomplete: bool = True,
            train_sos: bool = False,
            num_proc: int = 4,
    ):
        step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < step_size <= seq_len
        if not concatenate and not pad_incomplete:
            warnings.warn(
                'Sequence concatenation and padding are both disabled. The model will see very few <eos> tokens.'
            )
        column = 'indices'
        tokenizer = parent.tokenizer
        sos_index = tokenizer[tokenizer.sos_token]
        eos_index = tokenizer[tokenizer.eos_token]
        pad_index = tokenizer[tokenizer.pad_token]
        dataset = self._generate(parent, column, seq_len, step_size, concatenate, pad_incomplete, train_sos, num_proc,
            sos_index, eos_index, pad_index)
        super().__init__(column=column, parent=parent, dataset=dataset)
        self.concatenate = concatenate
        self.train_sos = train_sos
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index

    @staticmethod
    def _generate(parent, column, seq_len, step_size, concatenate, pad_incomplete, train_sos, num_proc,
            sos_index, eos_index, pad_index):
        # When train_sos is true and padding is enabled, sos will be added after eos, before regular padding,
        # thus training the model to start a new sequence after finishing a previous one.
        def generate():
            sos = np.array([sos_index])
            eos = np.array([eos_index])
            padding = np.array([pad_index]).repeat(seq_len)
            sos_pad = np.array([sos_index])

            reader = iter(parent)
            buffer = np.array([], dtype=int)
            window = seq_len + 1
            add_sos = train_sos

            while True:
                if len(buffer) < window:
                    if concatenate:
                        try:
                            buffer = np.concatenate([buffer, sos, next(reader), eos])
                            continue
                        except StopIteration:
                            pass

                    buf_thresh = 1 - add_sos  # require trainable tokens in buffer (1 if training sos, 2 otherwise)
                    # can add sos if pad=true or exactly token left
                    if (pad_incomplete or len(buffer) == window - add_sos) \
                            and len(buffer) > buf_thresh and buffer[buf_thresh] != pad_index:
                        buffer = np.concatenate([buffer, sos_pad[:add_sos], padding[:window - len(buffer) - add_sos]])
                        add_sos = False  # disable until next read
                    else:
                        try:
                            buffer = np.concatenate([sos, next(reader), eos])
                            add_sos = train_sos  # reset
                            continue
                        except StopIteration:
                            break

                yield {column: buffer[:window]}
                buffer = buffer[step_size:]

        dataset = hf_datasets.Dataset.from_generator(generate, num_proc=num_proc, split=parent.parent.split)
        return dataset.with_format('numpy')

    def _compute_mask(self, x: torch.Tensor):
        sos_indices = (x == self.sos_index).nonzero().squeeze(1)
        if sos_indices.size(0) == 0:
            return torch.ones((x.size(0),) * 2, dtype=torch.bool).tril()
        lengths = sos_indices[1:] - sos_indices[:-1]
        init = sos_indices[:1]
        remainder = x.size(0) - sos_indices[-1:]
        lengths = torch.cat([init, lengths, remainder])
        blocks = [torch.ones(l, l, dtype=torch.bool).tril() for l in lengths if l]
        return torch.block_diag(*blocks)

    def _get_xym(self, indices):
        indices = torch.from_numpy(indices)
        x, y = indices[:-1], indices[1:]
        if self.concatenate and not self.train_sos:  # unwanted target sos may be present only if concatenate is enabled
            y = torch.where(y == self.sos_index, self.pad_index, y)
        return (x, y, self._compute_mask(x)) if self.concatenate else (x, y)

    def __getitem__(self, idx):
        indices = self.dataset[idx][self.column]
        return self._get_xym(indices)


    def __iter__(self):
        for batch in self.dataset.iter(1000):
            for indices in batch[self.column]:
                yield self._get_xym(indices)
