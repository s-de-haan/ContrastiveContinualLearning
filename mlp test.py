import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin
from avalanche.benchmarks.utils import AvalancheDataset

import models
import losses

from avalanche.models import SimpleMLP

class ClassifierPlugin(SupervisedPlugin):
	def __init__(self):
		'''Add a new dimension to our filter layer after each task is trained'''
		super().__init__()
	
	def after_train_dataset_adaptation(self, strategy):
		print(dir(strategy))


class SupConFilterPlugin(SupervisedPlugin):
	def __init__(self):
		'''Add a new dimension to our filter layer after each task is trained'''
		super().__init__()
	
	def after_training_exp(self, strategy):
		training_set, labels, _ = strategy.adapted_dataset[:]
		training_set = training_set.to(device)
		with torch.no_grad():
			r = strategy.model.encoder(training_set)
			z_filtered = strategy.model.filter(strategy.model.projector(r))
			principle_direction = torch.svd(z_filtered).V[:,0]
			strategy.model.filter.append(principle_direction)
		# strategy.model.train_classifier(r, labels)

	# def after_eval_forward(self, strategy):
	# 	print(strategy.model)

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
model = models.SupervisedContrastiveLearner()

# CL Benchmark Creation
split_mnist = SplitMNIST(n_experiences=5, shuffle=False)
train_stream = split_mnist.train_stream
test_stream = split_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
classifier_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = losses.SupConLoss()
classes_order = train_stream.benchmark.classes_order

# train and test loop over the stream of experiences
results = []

class EvenOddCrossEntropy(CrossEntropyLoss):
	def __init__(self, class_order, **kwargs) -> None:
		self.class_order = class_order
		super(EvenOddCrossEntropy, self).__init__(**kwargs)

	def forward(self, input, target):
		target %= 2
		return super(EvenOddCrossEntropy, self).forward(input, target)

# Continual learning strategy
cl_strategy = Naive(model, optimizer, criterion, train_mb_size=128, train_epochs=1, eval_mb_size=128, device=device, plugins=[SupConFilterPlugin()])
classifier_strategy = Naive(model.classifier, classifier_optimizer, EvenOddCrossEntropy(classes_order), train_mb_size=128, train_epochs=1, eval_mb_size=128, device=device)

for task_num, train_exp in enumerate(train_stream):
	# train_exp.dataset.targets = (torch.tensor(train_exp.dataset.targets)==classes_order[2*task_num+1]).long()
	train_exp.dataset.targets = train_exp.dataset.targets % 2
	cl_strategy.train(train_exp)
	train_exp.dataset.transform = lambda x: model.encoder(x)
	classifier_strategy.train(train_exp)

	for test_exp in test_stream[:task_num+1]:
		# test_exp.dataset.targets = (torch.tensor(test_exp.dataset.targets)==classes_order[2*task_num+1]).long()
		test_exp.dataset.targets = test_exp.dataset.targets % 2
		test_exp.dataset.transform = lambda x: model.encoder(x)
		results.append(classifier_strategy.eval(test_exp))

# print(results)
print([res['Top1_Acc_Stream/eval_phase/test_stream/Task000'] for res in results])