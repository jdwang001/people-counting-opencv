import logging
import time
import os
from logging.handlers import RotatingFileHandler


class Log(object):

    def __init__(self, logger=None, log_cate='search'):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        self.log_time = time.strftime("%Y_%m_%d")
        file_dir =  '/tmp/counter'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'
        # print(self.log_name)

        #fh = logging.FileHandler(self.log_name, 'a')  # 追加模式  这个是python2的
        #fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        fh = RotatingFileHandler(self.log_name, 'a', 10485760, 10, encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.INFO)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        #  添加下面一句，在记录日志之后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
        ch.close()

    def getlog(self):
        return self.logger