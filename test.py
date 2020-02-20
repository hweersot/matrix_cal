import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.mlp import mlp
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import matplotlib.pyplot as plt


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = mlp(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    inverse_w=sess.run(tf.get_default_graph().get_tensor_by_name('weight1:0'))
    # here you train your model
    for i in range(len(inverse_w)):
        inverse_w[i]=1/inverse_w[i]
    
    print('층별 가격 비율은 \n')
    ratio=[]
    for i in range(30):
        ratio.append(float(inverse_w[i]/inverse_w[0]))
#    scale=['~20','20~30','30~40','40~50','50~60','60~70','70~80','80~90','90~100','100~120','120~140','140~160','160~180','180~200','200~220','220~']
    floor=[i for i in range(2,32)]
#    plt.bar(scale,ratio)
#    plt.xticks(rotation=90)
    plt.ylim(0,3)
    print(ratio)
    plt.plot(floor,ratio,label='ppa')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
