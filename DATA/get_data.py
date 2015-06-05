import sys

with open(sys.argv[1], 'r') as f:
	for line in f:
		data = line.strip().split(',')
		str = '\t'.join(data[:-1])
		str += '\t' + '0' if data[-1] == 'g' else '\t' + '1'
		print str
