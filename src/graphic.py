import os
import matplotlib.pyplot as plt



def draw_curve(x_epoch, y_loss, y_err, output_name):
    #x_epoch.append(current_epoch)
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    ax0.plot(x_epoch, y_loss['train'], 'b-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'r-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'b-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'r-', label='val')
    ax0.legend()
    ax1.legend()
    fig.savefig(os.path.join('./lossGraphs', output_name))