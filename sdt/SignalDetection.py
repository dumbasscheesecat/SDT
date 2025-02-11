from scipy.stats import norm

class SignalDetection:
	def __init__(self, hits, misses, false_alarms, correct_rejections):
		self.hits = hits
		self.misses = misses
		self.false_alarms = false_alarms
		self.correct_rejections = correct_rejections

	def hit_rate(self):
		return (self.hits) / (self.hits + self.misses) #gpt suggests .5 and 1 for edge cases of small samples

	def false_alarm_rate(self):
		return (self.false_alarms) / (self.false_alarms + self.correct_rejections)

	def d_prime(self):
		return norm.ppf(self.hit_rate()) - norm.ppf(self.false_alarm_rate()) #norm ppf found online on overflow

	def criterion(self):
		return -0.5 * (norm.ppf(self.hit_rate()) + norm.ppf(self.false_alarm_rate()))
