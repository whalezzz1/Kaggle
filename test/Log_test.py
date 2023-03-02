import argparse


# log string
def log_string(Log, string):
    Log.write(string + '\n')  # Python File write() 方法语法如下：fileObject.write( [ str ])
    Log.flush()  # file.flush()刷新文件内部缓冲，直接把内部缓冲区的数据立刻写入文件, 而不是被动地等待输出缓冲区写入。
    print(string)


parser = argparse.ArgumentParser()
parser.add_argument('--a', type=int, default=3, help='第一个参数')
parser.add_argument('--b', type=float, default=3.644, help='第二个参数')
parser.add_argument('--log_file',
                    default=r'D:\pycharm_project\traffic_others\GMAN-PyTorch-master\GMAN-PyTorch-master\test\log1',
                    help='log文件生成路径')
args = parser.parse_args()


log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
log_string(log, 'data loaded!')
log_string(log, 'compiling model...')
