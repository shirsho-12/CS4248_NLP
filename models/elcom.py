import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TeacherModel():
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, x):
        return self.model.encode(x)
    
    def __call__(self, x):
        return self.encode(x)

class StudentModel():
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def forward(self, x):
        return self.model.encode(x)
    
    def __call__(self, x):
        return self.forward(x)

# class ELCoM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.teacher = TeacherModel()
#         self.student = StudentModel()
    
#     def forward(self, x, y):
#         teacher_en = self.teacher(x)
#         student_en = self.student(x)
#         student_em = self.student(y)
#         return teacher_en, student_en, student_em

    # def __call__(self, x, y):
    #     return self.forward(x, y)
