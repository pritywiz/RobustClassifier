import os
import csv
import time

class LogWriter(object):
    def __init__(self, log_dir, filename):
        self.log_dir = log_dir
        self.filename = os.path.join(log_dir, filename)
        self.log_file = open(self.filename, 'w')

    def __call__(self, data):
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=['time'] + list(data.keys()))
        self.log_writer.writeheader()

    def writeLog(self, data):
        data['time'] = time.time()
        self.log_writer.writerow(data)

    def flushLog(self):
        self.log_file.flush()
        os.fsync(self.log_file.fileno())

    def close(self):
        self.log_file.close()

