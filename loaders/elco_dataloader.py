import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict, Tuple
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
    Class for loading the Elco dataset.
    The dataset is a csv file with the following format:
    label, emojis separated by [EM], list of emojis
"""
class ElcoDataset(Dataset): 
    """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Notes:
            Each label has 7,8, or 9 rows, each row has a different list of emojis
            Emoji and emoji_list columns hold the same information, but in different formats
    """
    def __init__(self, data: Dict, labels: List, transform=None): 
        self.transform = transform
        self.labels = labels
        self.data = data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, str]:
        label = self.labels[idx]
        data = self.data[label]
        
        for key, values in data.items():
            arr = []
            if isinstance(values, list):
                for value in values:
                    arr.append(value)
                data[key] = arr
            else:
                data[key] = [values]
            data[key].append(" ".join(data[key]))

        if self.transform:
            data = self.transform(data)
        return data, label

    def get_data(self) -> Dict:
        return self.data
    
    def get_labels(self) -> List:
        return self.labels

class ElcoDataLoader:
    def __init__(self, csv_file: str, batch_size: Optional[int]=1, 
                 shuffle: Optional[bool] =True, 
                 num_workers: Optional[int]=0, 
                 transform=None, test_size=0.2):
        self.train, self.test = ElcoDataset(csv_file, transform=transform, 
                                            test_size=test_size).get_data()
        self.train_loader = DataLoader(self.train, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, 
                                      shuffle=shuffle, num_workers=num_workers)
    
    def get_dataset(self) -> ElcoDataset:
        return self.dataset
    
    def get_labels(self) -> List:
        return self.dataset.get_labels()
    
    def get_data(self) -> Dict:
        return self.dataset.get_data()
    
    def get_data_by_label(self, label) -> dict:
        return self.dataset.get_data()[label]

def get_loaders(csv_file: str, batch_size=1, shuffle=True, num_workers=0, 
                transform=None, test_size=0.2) -> Tuple[DataLoader, DataLoader]:
    """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            
        Returns:
            train_loader, test_loader
        
        Notes:
            Each emoji in the dataloader is a tuple 
            The ranking of the emojis is an assumption following the csv file
        e.g. 
        [{
            0: [('shorts',), ('electric plug',)],
            1: [('pinching hand',), ('light bulb',), ('high voltage',)],
            2: [('cross mark',), ('pencil',), ('axe',), ('wrench',)],
            3: [('cross mark button',), ('potable water',)],
            4: [('fast down button',),
            ('coat',),
            ('woman’s clothes',),
            ('t-shirt',),
            ('jeans',)],
            5: [('double exclamation mark',), ('warning',), ('bread',)],
            6: [('hut',), ('cross mark',)],
            7: [('pinching hand',), ('package',)]
            },
        ('short supply',)]
    """
    df = pd.read_csv(csv_file)
    labels = df["English"].unique()
    train_data, test_data = {}, {}
    train_labels, test_labels = train_test_split(labels, test_size=test_size)
    for label in train_labels:
        rank_dict = {}
        sentences = df.loc[df["English"] == label, "Emoji"].apply(
            lambda x: list(filter(None, x.split("[EM]")))).tolist()
        
        for i in range(len(sentences)):
            rank_dict[i] = sentences[i]

        train_data[label] = rank_dict
    for label in test_labels:
        rank_dict = {}
        sentences = df.loc[df["English"] == label, "Emoji"].apply(
            lambda x: list(filter(None, x.split("[EM]")))).tolist()
        for i in range(len(sentences)):
            rank_dict[i] = sentences[i]

        test_data[label] = rank_dict
    train_data = ElcoDataset(train_data, train_labels, transform=transform)
    test_data = ElcoDataset(test_data, test_labels, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader
