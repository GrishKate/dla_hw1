'''
        specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(20),
            torchaudio.transforms.TimeMasking(100),
        )
        '''