import numpy as np
from scipy.stats import norm

class SignalDetection:
	def __init__(self, hits, misses, falseAlarms, correctRejections, totalTrials):
		self.hits = hits
		self.misses =  misses
		self.falseAlarms = falseAlarms
		self.correctRejections = correctRejections
		self.totalTrials = totalTrials

	def hit_rate(self):
		return (self.hits) / (self.hits + self.misses)

	def false_alarm_rate(self):
		return (self.falseAlarms) / (self.falseAlarms + self.correctRejections)

#creates an inverse phi function
	def invPhi(self, x):
		#this if statement ensures the returnd value is not infinite (Assisted by DeepSeek)
		if x == 0:
			x = 0.5 / self.totalTrials
		elif x == 1:
			x = 1 - 0.5 / self.totalTrials
		return norm.ppf(x)

	def d_prime(self):
		return SignalDetection.invPhi(self.hits) - SignalDetection.invPhi(self.falseAlarms)

	def criterion(self):
		return -0.5 * (SignalDetection.invPhi(self.hits)+ SignalDetection.invPhi(self.falseAlarms))
