import numpy as np

class SignalDetection:
	def __init__(self, hits, misses, falseAlarms, correctRejections):
		self.hits = hits
		self.misses =  misses
		self.falseAlarms = falseAlarms
		self.correctRejections = correctRejections

	def hit_rate(self):
		return (self.hits) / (self.hits + self.misses)

	def false_alarm_rate(self):
		return (self.falseAlarms) / (self.falseAlarms + self.correctRejections)

	def d_prime(self):

	def criterion(self):
