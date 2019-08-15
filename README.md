# sensor
传感器的分类和提取三元组内容



classifier:
    ELMO文件：用于对15种传感器分类；
    process_data文件：下是对原始数据的处理；
    stacking_classifier文件：是利用stacking思想（机器学习的算法）对15种类别进行分类。
最后的结果：利用elmo提取的词向量，然后下游任务利用残差网络（residual block）和attention能够提高分类的效果。其中F1的值为：0.93，recall: 0.95。
           利用stacking_classifier，其结果F1的值为：0.85， 
