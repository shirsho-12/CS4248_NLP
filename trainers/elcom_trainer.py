import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from itertools import chain

from metrics.similarity_metrics import CosineSimilarity
from torch.autograd import Variable
from itertools import chain

class ELCoMTrainer:
    def __init__(self, teacher_model, student_model, optim, lr, accuracy_metric):
        self.teacher = teacher_model
        self.student = student_model
        # self.teacher_optimizer = optim(self.teacher.model._first_module().parameters(), lr=lr)
        self.student_optimizer = optim(self.student.model._first_module().parameters(), lr=lr)
        self.accuracy_metric = accuracy_metric
        self.optim = optim
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity()
        self.similarity = CosineSimilarity()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_loader, epochs):
        # self.teacher.train()
        # self.teacher.to(self.device)
        # self.student.train()
        # self.student.to(self.device)
        total_loss = 0
        for epoch in range(epochs):
            for x, y in tqdm(train_loader):
                teacher_en = Variable(self.teacher(x), requires_grad=True).mean(axis=0).unsqueeze(0)
                student_en = Variable(self.student(x), requires_grad=True).mean(axis=0).unsqueeze(0)
                student_em = Variable(self.student(y), requires_grad=True)
                loss = self.mse(student_en, teacher_en) + self.mse(student_em, teacher_en)
                + self.cosine(student_en, student_em).item()
                # self.teacher_optimizer.zero_grad()
                self.student_optimizer.zero_grad()
                loss.backward()
                # self.teacher_optimizer.step()
                self.student_optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1} loss: {total_loss / len(train_loader)}")
        print("Training finished")
    
    def test(self, test_loader):
        
        accuracy = 0
        for x, y in tqdm(test_loader):
            y = y[0]
            rank_scores = []
            y_score = self.student(y)
            for rank, sentences in x.items():
                sentences = list(chain(*sentences))
                embeddings = self.student(sentences).mean(axis=0) # simple mean
                rank_scores.append((self.similarity(embeddings, y_score).cpu().detach().numpy()[0], rank))
            sorted_rank_scores = sorted(rank_scores, key=lambda x: x[0], reverse=True)
            accuracy += self.accuracy_metric(sorted_rank_scores)
        return accuracy / len(test_loader)
