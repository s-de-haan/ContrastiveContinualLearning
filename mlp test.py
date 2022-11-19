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
split_mnist = SplitMNIST(n_experiences=5)
train_stream = split_mnist.train_stream
test_stream = split_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
classifier_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = losses.SupConLoss()

# Continual learning strategy
cl_strategy = Naive(model, optimizer, criterion, train_mb_size=128, train_epochs=10, eval_mb_size=128, device=device, plugins=[SupConFilterPlugin()])
classifier_strategy = Naive(model.classifier, classifier_optimizer, CrossEntropyLoss(), train_mb_size=128, train_epochs=2, eval_mb_size=128, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
	cl_strategy.train(train_exp)
	train_exp.dataset.transform = lambda x: model.encoder(x)
	test_stream.dataset.transform = lambda x: model.encoder(x)
	classifier_strategy.train(train_exp)

	results.append(classifier_strategy.eval(test_stream))

# print(results)
print([res['Top1_Acc_Stream/eval_phase/test_stream/Task000'] for res in results])